"""
exp1_l96/methods.py
===================
Samplers and evaluation for the Lorenz '96 experiment (the model and
experiment configuration are described in l96.py).

BENCHMARK PROVENANCE. The configuration -- model,
truth -- follows the stochastic Lorenz '96
benchmark of Pacchiardi, Khoo & Dutta, "Generalized Bayesian
likelihood-free inference using scoring rules estimators"
(arXiv:2104.03889), the setting the paper's Section 3.1 evaluates
against. Everything needed to read this code is stated locally.

  df_etld_reflect : ours, DEFAULT. Runs in natural theta space with the
                    uniform box prior enforced by REFLECTION at the faces
                    (correct Langevin boundary treatment for a uniform
                    prior). The statistical linearization sees the raw
                    simulator -- no squashing map, so the appendix
                    study's curvature-bias mechanism is never fed by a
                    reparameterization.
  smc_abc         : SMC-ABC with adaptive tolerance, Euclidean distance
                    on the raw dataset or on the 6 summary statistics.
  adsgld          : gradient-based baseline. RMSprop-preconditioned SGLD
                    with EXACT forward sensitivities through the RK4
                    simulator, on the energy score (benchmark
                    configuration) or on DF-ETLD's kernel-score target.
  energy_score_per_timestep : posterior-predictive evaluation metric;
                    supports n observed series (data term averaged over
                    them).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.dfetld import run_dfetld
from common.mmd import median_heuristic_from_sims

import l96


# ----------------------------------------------------------------------
# DF-ETLD (default)
# ----------------------------------------------------------------------
def df_etld_reflect(y_obs, x0, gamma, rng, M=100, Jsim=10, w=30.0,
                    dt=2e-3, n_steps=500, burn_in=250, thin=5):
    """Uniform box prior: zero prior gradient inside the box, uniform
    initialization, reflection after every step.

    LEARNING RATE CONVENTION. The target is the scoring-rule posterior

        pi(theta | y_n) propto pi(theta) exp(-w sum_{i=1}^n S_k(P_theta, y_i)),

    with a SUM over the n observed series, so the posterior concentrates as n
    grows. For the kernel score this is exactly n * MMD^2 against the
    empirical measure, up to a theta-independent constant (verified to
    machine precision), so the sampler still uses the averaged MMD^2 drift
    and simply sets

        beta = w * n_obs.

    w is on the same per-observation scale as the benchmark's
    energy-score weight.
    Returns (samples in theta space, simulator calls, diagnostics)."""
    steps, K = l96.n_latents()
    calls = {"n": 0}
    beta = w * np.atleast_2d(y_obs).shape[0]      # sum-over-observations

    def simulate_t(th, U):
        calls["n"] += th.shape[0] * U.shape[0]
        return l96.simulate(th, x0, U)

    samples, diag = run_dfetld(
        simulate_t, np.atleast_2d(y_obs),
        np.zeros(l96.D_PAR), np.ones(l96.D_PAR), M=M, J=Jsim,
        D=l96.D_PAR, beta=beta, gamma=gamma, dt=dt, n_steps=n_steps,
        burn_in=burn_in, thin=thin, rng=rng,
        draw_latents=lambda r: r.standard_normal((Jsim, steps, K)),
        grad_log_prior=lambda th: np.zeros_like(th),
        sample_prior=lambda r, m: l96.sample_box(r, m),
        project=l96.reflect_box)
    return samples, calls["n"], diag


# ----------------------------------------------------------------------
# SMC-ABC, Euclidean distance on raw data
# ----------------------------------------------------------------------
def smc_abc(y_obs, x0, rng, n_pop=1000, n_gen=25, m=10,
            quantile=0.5, budget=None, summaries=False):
    """
    Adaptive SMC-ABC on the uniform box prior. Defaults are the
    benchmark configuration: n_pop = 1000 particles, m = 10 simulations
    per parameter, 25 generations -> 250,000 simulator calls. `budget`,
    if given, is an additional early stop. Distance:
    mean_{j,i}||x_j - y_i|| over m simulated and n observed series.

    summaries=False (default): the distance runs on the raw 360-dim
    series. summaries=True: the same algorithm and budget with the
    distance on the 6 summary statistics (l96.summary_stats) -- the
    second ABC row of the paper's Table 1.
    """
    lo, hi = l96.BOX_LO, l96.BOX_HI
    D = l96.D_PAR
    steps, K = l96.n_latents()
    calls = {"n": 0}
    Y2 = np.atleast_2d(y_obs)
    Yd = l96.summary_stats(Y2) if summaries else Y2      # (n, 6) or (n,360)

    def dist(theta):
        calls["n"] += theta.shape[0] * m
        U = rng.standard_normal((m, steps, K))
        X = l96.simulate(theta, x0, U)                 # (n_theta, m, 360)
        Xd = l96.summary_stats(X) if summaries else X
        d = np.linalg.norm(Xd[:, :, None, :] - Yd[None, None, :, :],
                           axis=-1)
        return d.mean(axis=(1, 2))

    theta = l96.sample_box(rng, n_pop)
    d = dist(theta)
    w = np.full(n_pop, 1.0 / n_pop)

    for g in range(n_gen - 1):
        if budget is not None and calls["n"] + n_pop * m > budget:
            break
        eps = np.quantile(d, quantile)
        keep = d <= eps
        th_k, w_k = theta[keep], w[keep]
        w_k = w_k / w_k.sum()
        cov = 2.0 * np.cov(th_k.T, aweights=w_k) + 1e-9 * np.eye(D)
        L = np.linalg.cholesky(cov)
        idx = rng.choice(len(th_k), n_pop, p=w_k)
        prop = th_k[idx] + rng.standard_normal((n_pop, D)) @ L.T
        bad = np.any((prop < lo) | (prop > hi), axis=1)
        while bad.any():
            j = rng.choice(len(th_k), bad.sum(), p=w_k)
            prop[bad] = th_k[j] + rng.standard_normal((bad.sum(), D)) @ L.T
            bad = np.any((prop < lo) | (prop > hi), axis=1)
        theta = prop
        d = dist(theta)
        diff = theta[:, None, :] - th_k[None, :, :]
        sol = np.linalg.solve(cov, diff.reshape(-1, D).T).T.reshape(
            len(theta), len(th_k), D)
        lq = -0.5 * np.einsum("nkd,nkd->nk", diff, sol)
        lq = np.log((np.exp(lq - lq.max(1, keepdims=True)) * w_k).sum(1)
                    + 1e-300) + lq.max(1)
        w = np.exp(-(lq - lq.min()))
        w = w / w.sum()
    keep = d <= np.quantile(d, quantile)
    return theta[keep], w[keep] / w[keep].sum(), calls["n"]


# ----------------------------------------------------------------------
# adSGLD: gradient-based baseline
# ----------------------------------------------------------------------
W_ES = 3.4882              # the benchmark's energy-score weight
                           # (estimated against synthetic likelihood for
                           # the 4-parameter AR(1) run; reused unchanged
                           # in the 3-parameter mode)


def adsgld(y_obs, x0, rng, m=200, w=W_ES, n_steps=2500, step=2e-2,
           alpha=0.99, lam=1e-6, burn_frac=0.5, thin=5, n_chains=1,
           loss="es", gamma=None):
    """
    Gradient-based baseline: RMSprop-preconditioned SGLD on the
    energy-score generalized posterior

        pi(theta | y) prop. prior(theta) * exp(-w * ES(P_theta, y)),

    with EXACT forward sensitivities through the RK4/AR(1) simulator
    (l96.simulate_rk4_sens, finite-difference verified to ~1e-9).
    All methods share the same prior measure.

    Defaults follow the benchmark configuration, and m = 200 simulations per
    gradient step; the benchmark run used 20000 steps = 4,000,000
    simulator calls. n_steps is left to the caller so the baseline can
    be run either budget-matched or at full length.

    ES estimator with n observed series, m simulations:
        ES  = (2/(m n)) sum_{j,i} ||x_j - y_i||
              - (1/(m(m-1))) sum_{j != l} ||x_j - x_l||
        dES/dx_j = (2/m) mean_i u(x_j - y_i)
                   - (2/(m(m-1))) sum_{l != j} u(x_j - x_l)
    Both pairwise sums are formed from (m, .) kernels without
    materializing any (m, m, 360) tensor.

    loss = "es" (default): the energy-score posterior (the benchmark
    baseline).
    loss = "kernel": the KERNEL-score posterior with bandwidth `gamma` --
    exp(-w sum_i S_k(P, y_i)), which for the Gaussian kernel is EXACTLY
    the generalized posterior DF-ETLD samples (S_k sums to n*MMD^2 up to
    a theta-free constant; verified numerically earlier). Running adSGLD
    with loss="kernel", the same w and the same gamma as DF-ETLD makes
    the two samplers target the IDENTICAL distribution, so their
    disagreement isolates the SAMPLER effect from the LOSS effect, which
    the default ES-vs-MMD comparison confounds.

    Returns (samples in theta space, simulator calls, diagnostics).
    """
    if loss == "kernel" and gamma is None:
        raise ValueError("kernel loss needs the bandwidth gamma")
    Y2 = np.atleast_2d(y_obs)
    n_obs = Y2.shape[0]
    steps_lat, K = l96.n_latents()
    burn = int(burn_frac * n_steps)
    kept, calls, tame, chain_means = [], 0, 0.0, []

    for _ in range(n_chains):
        theta = l96.sample_box(rng, 1)[0]
        V = np.zeros(l96.D_PAR)
        kept_c = []
        for k in range(n_steps):
            U = rng.standard_normal((m, steps_lat, K))
            X, S = l96.simulate_rk4_sens(theta, x0, U)
            calls += m

            d2y = (np.sum(X ** 2, 1)[:, None] + np.sum(Y2 ** 2, 1)[None, :]
                   - 2.0 * X @ Y2.T)
            d2x = (np.sum(X ** 2, 1)[:, None] + np.sum(X ** 2, 1)[None, :]
                   - 2.0 * X @ X.T)
            if loss == "es":
                Wy = 1.0 / (np.sqrt(np.maximum(d2y, 0.0)) + 1e-12)
                gy = X * Wy.mean(1)[:, None] - (Wy @ Y2) / n_obs
                Wx = 1.0 / (np.sqrt(np.maximum(d2x, 0.0)) + 1e-12)
                np.fill_diagonal(Wx, 0.0)
                gxx = X * Wx.sum(1)[:, None] - Wx @ X
                dES_dx = (2.0 / m) * gy - (2.0 / (m * (m - 1))) * gxx
            else:
                # kernel score, summed over the n observed series:
                #   L = n E k(X,X') - 2 sum_i E k(X,y_i)   (+ const)
                # with grad_x k(x,y) = -(x - y)/gamma^2 k(x,y)
                Ky = np.exp(-np.maximum(d2y, 0.0) / (2 * gamma ** 2))
                Kx = np.exp(-np.maximum(d2x, 0.0) / (2 * gamma ** 2))
                np.fill_diagonal(Kx, 0.0)
                g2 = gamma ** 2
                # d/dx_j of n * (1/(m(m-1))) sum_{a!=b} k(x_a,x_b)
                gxx = -(X * Kx.sum(1)[:, None] - Kx @ X) / g2
                gxx *= 2.0 * n_obs / (m * (m - 1))
                # d/dx_j of -2 * (1/m) sum_i k(x_j, y_i)
                gy = (X * Ky.sum(1)[:, None] - Ky @ Y2) / g2
                gy *= 2.0 / m
                dES_dx = gxx + gy
            gbar = w * np.einsum("md,mdp->p", dES_dx, S)   # prior grad = 0

            V = alpha * V + (1.0 - alpha) * gbar ** 2
            G = 1.0 / (np.sqrt(V) + lam)
            drift = 0.5 * step * G * gbar
            if k >= burn:
                tame = max(tame, float(np.max(np.abs(drift))))
            theta = theta - drift + np.sqrt(step * G) * \
                rng.standard_normal(l96.D_PAR)
            theta = l96.reflect_box(theta)

            if k >= burn and (k - burn) % thin == 0:
                kept_c.append(theta.copy())
        kept_c = np.array(kept_c)
        chain_means.append(kept_c.mean(0))
        kept.append(kept_c)

    return (np.concatenate(kept), calls,
            {"tame_stationary": tame,
             "chain_means": np.array(chain_means)})


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
def energy_score_per_timestep(pred, y_obs):
    """
    pred : (S, N_REC, K) posterior-predictive trajectories
    y_obs: (n, N_REC * K) observed series (n >= 1)
    Returns (N_REC,) energy scores; data term averaged over the n series.
    (Index 0 is the shared IC column -> identically ~0.)
    """
    S, T, K = pred.shape
    y = np.atleast_2d(y_obs).reshape(-1, T, K)
    es = np.empty(T)
    for t in range(T):
        x = pred[:, t]                                   # (S, K)
        t1 = 2.0 * np.linalg.norm(x[:, None] - y[None, :, t],
                                  axis=2).mean()
        dd = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
        t2 = dd.sum() / (S * (S - 1))
        es[t] = t1 - t2
    return es


def posterior_predictive(theta_draws, x0, rng):
    """Fresh-noise simulations at posterior draws: (S, N_REC, K)."""
    steps, K = l96.n_latents()
    out = []
    for th in theta_draws:
        U = rng.standard_normal((1, steps, K))
        out.append(l96.simulate(th[None], x0, U)[0, 0].reshape(steps, K))
    return np.stack(out)
