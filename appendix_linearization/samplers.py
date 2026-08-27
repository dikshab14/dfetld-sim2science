"""
samplers.py
===========
Rungs 2 and 3 of the comparison ladder. Both are Euler-Maruyama
discretizations of the SAME interacting Langevin SDE (paper eqs. (15)-(17),
specialized to D = 1); the ONLY difference is how the MMD-gradient term in
the drift is computed:

  rung 2  aldi_exact_gradient : uses the closed-form  d/dtheta MMD^2  (M7)
                                -> no linearization anywhere
  rung 3  df_etld             : uses the statistical-linearization surrogate
                                g^m of paper eqs. (18)-(21), built from J
                                forward simulations per particle

so that (rung 3) - (rung 2), run with identical M, dt, beta, gamma, prior,
and initialization, isolates linearization bias + finite-J noise. Running
rung 3 additionally at large J shows the finite-J part is subdominant.

The common SDE (D = 1, so C_s is a scalar variance):

  d theta^m = [ C_s * grad_log_prior(theta^m)
                - beta * (MMD-gradient term)^m
                + ((D+1)/M) * (theta^m - theta_bar) ] ds
              + sqrt(2 C_s) dW^m                                    (S1)

* In rung 2 the MMD term is  C_s * grad_mmd2_exact(theta^m).
* In rung 3 the MMD term is  g^m  from eq. (7): substituting the
  statistical linearization C_s^{theta x_j} ~ C_s * dG/dtheta (paper
  Appendix A) into the empirical MMD gradient (4)-(5) replaces
  "C_s * gradient" directly by g^m -- no C_s^{-1} is ever formed.

* The (D+1)/M correction term and the multiplicative sqrt(2 C_s) noise are
  exactly the ALDI ingredients (Garbuno-Inigo, Nuesken, Reich 2020) that
  make the finite-M system affine invariant and target the right measure.

* Ensemble covariance: the paper defines C_s with a 1/M factor;
  we use 1/M, i.e. np.var(theta) without Bessel correction.
"""

import numpy as np

from model import (simulate, grad_log_prior, grad_mmd2_exact,
                   grad_x_gaussian_kernel)


# ----------------------------------------------------------------------
# shared machinery
# ----------------------------------------------------------------------
def _init_ensemble(M, m0, s0, rng):
    """Prior initialization (cfg.init_mode = "prior"), as in the paper."""
    return m0 + s0 * rng.standard_normal(M)


def _step_common(theta, drift_mmd, dt, beta, m0, s0, M, C_s, rng):
    """
    One Euler-Maruyama step of (S1), given the method-specific MMD drift term
    `drift_mmd` (an (M,) array standing for C_s * dMMD^2/dtheta or g^m).
    """
    D = 1
    theta_bar = theta.mean()
    drift = (C_s * grad_log_prior(theta, m0, s0)      # prior term
             - beta * drift_mmd                       # data-fit term
             + ((D + 1) / M) * (theta - theta_bar))   # finite-M correction

    tame_activity = float(np.max(dt * np.abs(drift)))  # diagnostic, see below
    drift = drift / (1.0 + dt * np.abs(drift))

    noise = np.sqrt(2.0 * max(C_s, 0.0) * dt) * rng.standard_normal(M)
    return theta + dt * drift + noise, tame_activity


def _collect(theta_traj, burn_in, thin):
    """Pool post-burn-in, thinned iterates across all particles."""
    kept = theta_traj[burn_in::thin]                  # (n_kept, M)
    return np.concatenate([snap for snap in kept])


# ----------------------------------------------------------------------
# rung 2: ALDI with the exact MMD gradient (no linearization)
# ----------------------------------------------------------------------
def aldi_exact_gradient(cfg, y, gamma, rng, theta0=None):
    """
    Interacting Langevin dynamics (S1) with the closed-form gradient (M7).
    Everything (M, dt, n_steps, prior, beta, gamma, init) matches df_etld.
    theta0: optional initial ensemble (warm start); prior draw if None.
    """
    theta = (_init_ensemble(cfg.M, cfg.m0, cfg.s0, rng)
             if theta0 is None else theta0.copy())
    traj = np.empty((cfg.n_steps, cfg.M))
    diag_minC, diag_tame = np.inf, 0.0

    for t in range(cfg.n_steps):
        C_s = float(np.var(theta))                    # 1/M convention
        diag_minC = min(diag_minC, C_s)
        # exact-gradient rung: MMD drift term = C_s * dMMD^2/dtheta
        drift_mmd = C_s * grad_mmd2_exact(theta, y, cfg.alpha,
                                          cfg.sigma, gamma)
        theta, tame = _step_common(theta, drift_mmd, cfg.dt, cfg.beta,
                                   cfg.m0, cfg.s0, cfg.M, C_s, rng)
        if t >= cfg.burn_in:      # taming at STATIONARITY is what matters:
            diag_tame = max(diag_tame, tame)   # if max dt|drift| is not
        traj[t] = theta                        # << 1 there, taming distorts
                                               # the invariant measure and dt
                                               # must be reduced.
    samples = _collect(traj, cfg.burn_in, cfg.thin)
    return samples, {"min_C_s": diag_minC, "tame_stationary": diag_tame,
                     "traj": traj}


# ----------------------------------------------------------------------
# rung 3: DF-ETLD (statistical linearization, simulator-derivative-free)
# ----------------------------------------------------------------------
def _df_etld_drift(theta, X, y, gamma, J, N):
    """
    The derivative-free surrogate g^m of paper eq. (7), for D = 1.

    Inputs
    ------
    theta : (M,)   particles theta^m
    X     : (M, J) simulated outputs x^{mj} = G_{theta^m}(u^j)
    y     : (N,)   observed data

    Construction (all quantities per paper Section 2):
      x_bar^j    = (1/M) sum_m x^{mj}                       (eq. (6))
      C^{theta x_j} = (1/M) sum_m (theta^m - theta_bar)(x^{mj} - x_bar^j)
                                                            (eq. (6))
      g^m = (2/(J(J-1))) sum_{j != l} C^{theta x_j} d_x k(x^{mj}, x^{ml})
          - (2/(J N))    sum_j sum_n  C^{theta x_j} d_x k(x^{mj}, y^n)
                                                            (eq. (7))
    Only forward simulations X and the ANALYTIC kernel gradient d_x k enter:
    no simulator Jacobian is ever touched.
    """
    M = len(theta)
    theta_dev = theta - theta.mean()                       # (M,)
    x_bar = X.mean(axis=0)                                  # (J,)
    # cross-covariances C^{theta x_j}, one scalar per j     (J,)
    C_tx = (theta_dev[:, None] * (X - x_bar[None, :])).mean(axis=0)

    # --- model-model kernel-gradient sums --------------------------------
    # d_x k(x^{mj}, x^{ml}) for all m, j, l:  (M, J, J)
    dk_xx = grad_x_gaussian_kernel(X[:, :, None], X[:, None, :], gamma)
    # zero out the diagonal j = l (unbiased U-statistic form of eq. (2))
    jj = np.arange(X.shape[1])
    dk_xx[:, jj, jj] = 0.0
    # sum over l, weight by C^{theta x_j}, sum over j
    term_xx = (C_tx[None, :] * dk_xx.sum(axis=2)).sum(axis=1)   # (M,)
    term_xx *= 2.0 / (X.shape[1] * (X.shape[1] - 1))

    # --- model-data kernel-gradient sums ---------------------------------
    # d_x k(x^{mj}, y^n) for all m, j, n:  (M, J, N)
    dk_xy = grad_x_gaussian_kernel(X[:, :, None], y[None, None, :], gamma)
    term_xy = (C_tx[None, :] * dk_xy.sum(axis=2)).sum(axis=1)   # (M,)
    term_xy *= 2.0 / (X.shape[1] * N)

    return term_xx - term_xy                                    # g^m, (M,)


def df_etld(cfg, y, gamma, rng, theta0=None):
    """
    Derivative-free ensemble transform Langevin dynamics, discretized by Euler-Maruyama.

    Common random numbers: the latent draws u^j are shared across particles
    within a step (as in the paper).
    """
    theta = (_init_ensemble(cfg.M, cfg.m0, cfg.s0, rng)
             if theta0 is None else theta0.copy())
    traj = np.empty((cfg.n_steps, cfg.M))
    diag_minC, diag_tame = np.inf, 0.0
    N = len(y)

    # This first draw is replaced on the first iteration; it is kept so the
    # random stream matches the runs that produced the committed outputs.
    U = rng.standard_normal(cfg.J)
    for t in range(cfg.n_steps):
        # fresh latents each step, shared across all particles
        U = rng.standard_normal(cfg.J)
        X = simulate(theta, cfg.alpha, cfg.sigma, U)  # (M, J) forward sims

        C_s = float(np.var(theta))
        diag_minC = min(diag_minC, C_s)
        drift_mmd = _df_etld_drift(theta, X, y, gamma, cfg.J, N)
        theta, tame = _step_common(theta, drift_mmd, cfg.dt, cfg.beta,
                                   cfg.m0, cfg.s0, cfg.M, C_s, rng)
        if t >= cfg.burn_in:
            diag_tame = max(diag_tame, tame)
        traj[t] = theta

    samples = _collect(traj, cfg.burn_in, cfg.thin)

    # diagnostic: average stationary ensemble spread tr(C_s)
    post_burn = traj[cfg.burn_in:]
    mean_C_stationary = float(np.mean(np.var(post_burn, axis=1)))
    return samples, {"min_C_s": diag_minC,
                     "tame_stationary": diag_tame,
                     "mean_C_stationary": mean_C_stationary,
                     "traj": traj}
