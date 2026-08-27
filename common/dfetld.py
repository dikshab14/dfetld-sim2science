"""
common/dfetld.py
================
General D-dimensional DF-ETLD sampler (paper Section 2), the production
version of what the appendix linearization study implements in 1-D. Works with ANY black-box simulator
exposed as

    simulate(theta, U) -> X          theta: (M, D) particle parameters
                                     U:     latent noise, shared across
                                            particles (common random numbers,
                                            as in the paper's setup)
                                     X:     (M, J, n) flattened outputs

No simulator derivative is ever requested.

Implemented objects, with their equation numbers in the paper:
  * ensemble mean/covariance C_s = (1/M) Theta'^T Theta'      (Sec. 2)
  * cross-covariances C_s^{theta x_j} per latent index j      eq. (6)
  * derivative-free MMD drift g^m via statistical
    linearization                                             eqs. (5)-(7)
  * interacting SDE with prior term, (D+1)/M correction, and
    multiplicative noise sqrt(2) C_s^{1/2} dW^m               eq. (8)
  * generalized ensemble square root C_s^{1/2} = Theta'^T / sqrt(M),
    avoiding any Cholesky factorization                       (Sec. 2)
  * tamed Euler-Maruyama discretization,
    the stabilization identified as necessary in the appendix-study
    validation runs;
    the max of dt*|drift| after burn-in is logged so one can verify taming
    is INACTIVE at stationarity (active taming would bias the invariant
    measure and calls for a smaller dt).

Memory note: the model-model kernel-gradient sums are computed WITHOUT
materializing the (M, J, J, n) tensor, via
    sum_l grad_x k(x_j, x_l) = -(x_j * sum_l K_jl - sum_l K_jl x_l)/gamma^2,
which needs only (M, J, J) kernel matrices and (M, J, n) outputs.
"""

import numpy as np


def _drift_gf(theta, X, y, gamma):
    """
    Derivative-free MMD drift g^m, eq. (7), vectorized over particles.
    theta: (M, D), X: (M, J, n), y: (N, n); gamma is the Gaussian-kernel
    bandwidth.  Returns (M, D).
    """
    M, J, n = X.shape
    N = y.shape[0]
    g = float(gamma)

    th_dev = theta - theta.mean(0)                       # (M, D)
    X_dev = X - X.mean(0)                                # (M, J, n)
    # C^{theta x_j}: (J, D, n)  --  eq. (6), one D x n block per j
    C_tx = np.einsum("md,mjn->jdn", th_dev, X_dev) / M

    d2 = (np.sum(X ** 2, 2)[:, :, None] + np.sum(X ** 2, 2)[:, None, :]
          - 2.0 * np.einsum("mjn,mln->mjl", X, X))       # (M, J, J)
    d2y = (np.sum(X ** 2, 2)[:, :, None] + np.sum(y ** 2, 1)[None, None, :]
           - 2.0 * np.einsum("mjn,Nn->mjN", X, y))       # (M, J, N)
    idx = np.arange(J)

    Kxx = np.exp(-d2 / (2 * g ** 2))
    Kxx[:, idx, idx] = 0.0                               # U-statistic: l != j
    S_xx = -(Kxx.sum(2)[:, :, None] * X
             - np.einsum("mjl,mln->mjn", Kxx, X)) / g ** 2
    Kxy = np.exp(-d2y / (2 * g ** 2))
    S_xy = -(Kxy.sum(2)[:, :, None] * X
             - np.einsum("mjN,Nn->mjn", Kxy, y)) / g ** 2

    term_xx = np.einsum("jdn,mjn->md", C_tx, S_xx) * (2.0 / (J * (J - 1)))
    term_xy = np.einsum("jdn,mjn->md", C_tx, S_xy) * (2.0 / (J * N))
    return term_xx - term_xy                             # g^m, eq. (7)


def run_dfetld(simulate, y, prior_mean, prior_std, *, M, J, D, beta, gamma,
               dt, n_steps, burn_in, thin, rng, draw_latents,
               grad_log_prior=None, sample_prior=None, project=None):
    """
    Run DF-ETLD; returns pooled samples (n_samples, D) and diagnostics.

    simulate(theta, U) -> (M, J, n); draw_latents(rng) -> U for one step
    (resampled every step, matching the paper).

    Non-Gaussian priors (e.g. exp3_sir's logit-space standard-logistic): pass
    grad_log_prior(theta) -> (M, D) and sample_prior(rng, M) -> (M, D);
    both default to the N(prior_mean, prior_std^2) product prior, keeping
    every existing caller unchanged.

    project(theta) -> theta, if given, is applied after every step --
    e.g. reflection into a box for a uniform prior with bounded support
    (reflecting boundary is the correct Langevin treatment there, and it
    keeps the statistical linearization in NATURAL space, avoiding the
    curvature-induced bias of running through a squashing map).
    """
    prior_mean = np.asarray(prior_mean, float)
    prior_std = np.asarray(prior_std, float)
    if grad_log_prior is None:
        grad_log_prior = lambda th: -(th - prior_mean) / prior_std ** 2
    if sample_prior is not None:
        theta = sample_prior(rng, M)
    else:
        theta = prior_mean + prior_std * rng.standard_normal((M, D))
    traj = np.empty((n_steps, M, D))
    diag = {"min_eig_C": np.inf, "tame_stationary": 0.0}

    for t in range(n_steps):
        U = draw_latents(rng)
        X = simulate(theta, U)                           # forward sims only
        th_dev = theta - theta.mean(0)
        C = th_dev.T @ th_dev / M                        # (D, D) covariance C_s
        diag["min_eig_C"] = min(diag["min_eig_C"],
                                float(np.linalg.eigvalsh(C).min()))

        g = _drift_gf(theta, X, y, gamma)
        drift = (grad_log_prior(theta) @ C               # C grad log prior
                 - beta * g                              # linearized MMD term
                 + ((D + 1) / M) * th_dev)               # finite-M correction
        tame = float(np.max(dt * np.abs(drift)))
        if t >= burn_in:
            diag["tame_stationary"] = max(diag["tame_stationary"], tame)
        drift = drift / (1.0 + dt * np.abs(drift))       # tamed EM

        # noise: sqrt(2 dt) C^{1/2} xi, with the generalized square root
        # C^{1/2} = Theta'^T/sqrt(M) (paper Sec. 2) -> per-particle M-dim xi
        Xi = rng.standard_normal((M, M))
        theta = theta + dt * drift + np.sqrt(2.0 * dt / M) * Xi @ th_dev
        if project is not None:
            theta = project(theta)
        traj[t] = theta

    kept = traj[burn_in::thin]                           # (n_kept, M, D)
    samples = kept.reshape(-1, D)
    diag["traj"] = traj
    return samples, diag
