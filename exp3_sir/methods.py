"""
exp3_sir/methods.py
========================
DF-ETLD, SMC-ABC and SGLD on the Kelly et al. (2024) Section 4.4
misspecified SIR benchmark.

All three methods share, by construction:
  * the identical prior (eta ~ U(0,0.5), beta | eta ~ U(eta,0.5)) --
    exactly;
  * the identical observed series and the identical six summary statistics;
  * the SAME standardisation of summaries, computed once from a
    prior-predictive batch and frozen. Kelly et al. standardise summaries
    unconditionally using the mean and sd of all simulated summaries in the
    training set; we do the same, from a fixed prior-predictive sample, so
    no method sees a different scaling;
  * for the two MMD-based methods, the same frozen median-heuristic
    bandwidth, so the discrepancy is identical and only the SAMPLER differs;
  * simulator-call counting, for the cost comparison.

SGLD note: the gradient is taken by central finite differences
under common random numbers (4 extra simulator evaluations per step at
d_theta = 2). This is a faithful stand-in for the *gradient* and is
GENEROUS to the baseline (it grants exact derivative access that DF-ETLD
never uses).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.dfetld import run_dfetld
from common.mmd import median_heuristic_from_sims

import sir_model as S


# ----------------------------------------------------------------------
# shared preprocessing
# ----------------------------------------------------------------------
def build_standardiser(rng, n_pp=200, chunk=25):
    """
    Prior-predictive summaries -> (mean, sd) used to standardise every
    summary vector, plus the frozen kernel bandwidth. Computed once.

    Each prior draw gets its OWN Brownian path (see predictive_summaries):
    sharing one path across all n_pp draws understates the prior-predictive
    spread of the noise-driven summaries by ~2.3x, which both mis-scales
    the standardisation and shrinks the median-heuristic bandwidth fitted
    to it.
    """
    th = S.sample_theta_prior(rng, n_pp)
    X = np.empty((n_pp, S.T_DAYS))
    for a in range(0, n_pp, chunk):
        blk = th[a:a + chunk]
        U = S.draw_latents(rng, len(blk))
        d = np.arange(len(blk))
        X[a:a + len(blk)] = S.simulate_prevalence(blk, U)[d, d]
    s = S.summaries(X)
    mu, sd = s.mean(0), np.maximum(s.std(0), 1e-8)
    gamma = median_heuristic_from_sims((s - mu) / sd)
    return mu, sd, gamma, n_pp


def std_summaries(x, mu, sd):
    return (S.summaries(x) - mu) / sd


# ----------------------------------------------------------------------
# 1. DF-ETLD  (ours)
# ----------------------------------------------------------------------
def df_etld(s_obs, mu, sd, gamma, rng, M=40, J=8, beta_temp=10.0, dt=2e-2,
            n_steps=600, burn_in=300, thin=2, prior="published"):
    """
    MMD-Bayes posterior over the six standardised summaries, sampled by the
    derivative-free ensemble transform. Returns theta samples (beta, eta),
    simulator-call count, and diagnostics.

    beta_temp = 10, CALIBRATED. Frozen here and applied unchanged to the misspecified
    observation.
    """
    calls = {"n": 0}

    def simulate_z(z, U):
        calls["n"] += z.shape[0] * U.shape[0]
        X = S.simulate_prevalence(S.z_to_theta(z), U)     # (M, J, T)
        return std_summaries(X, mu, sd)                   # (M, J, 6)

    z, diag = run_dfetld(
        simulate_z, s_obs[None], np.zeros(2), np.ones(2), M=M, J=J, D=2,
        beta=beta_temp, gamma=gamma, dt=dt, n_steps=n_steps,
        burn_in=burn_in, thin=thin, rng=rng,
        draw_latents=lambda r: S.draw_latents(r, J),
        grad_log_prior=(S.grad_log_prior_z if prior == "published"
                        else S.grad_log_prior_z_uniform),
        sample_prior=((lambda r, m: S.sample_z_prior(r, m))
                      if prior == "published"
                      else (lambda r, m: S.sample_z_prior_uniform(r, m))))
    return S.z_to_theta(z), calls["n"], diag


# ----------------------------------------------------------------------
# 2. SMC-ABC on the six summaries
# ----------------------------------------------------------------------
def smc_abc(s_obs, mu, sd, rng, n_pop=200, quantile=0.5, m=1, budget=None,
            max_gen=100000, tol_rel=0.005, patience=6,
            min_unique_frac=0.2, trace=None, prior="published"):
    """
    Adaptive ABC-SMC: Euclidean distance on the six standardised
    summaries, tolerance = `quantile` of current distances, Gaussian
    perturbation kernel at twice the weighted covariance, proposals
    rejected outside the prior triangle.

    STOPPING. Run to CONVERGENCE, not to a call budget. The tolerance
    falls geometrically at first and then plateaus: once the population
    can no longer get closer to the observation, further generations only
    shrink it around whichever particle the noise favoured -- the
    collapse the README warns about. The run therefore stops when EITHER

      * the tolerance has improved by less than `tol_rel` (relative) for
        `patience` consecutive generations -- it has converged; or
      * the population COLLAPSES, measured as the number of DISTINCT
        particles falling below `min_unique_frac` of the population.

    `budget`, if given, is an additional cap, but it is no longer the
    intended stop. `trace`, if a list, receives per-generation
    (calls, tolerance, ESS, weighted mean) for the convergence plot.
    """
    calls = {"n": 0}
    stalled, eps_prev = 0, np.inf
    stop_reason = "max_gen"

    def dist(theta, chunk=10):
        """
        Distance for each proposed theta, each under its OWN Brownian
        path. Passing one path to the whole population (which
        simulate_prevalence would broadcast) makes the entire distance
        landscape shift together from generation to generation, so the
        tolerance sequence is driven by which path was drawn rather than
        by the population improving. 
        """
        n = theta.shape[0]
        calls["n"] += n * m
        s = np.empty((n, 6))
        for a in range(0, n, chunk):
            blk = theta[a:a + chunk]
            acc = np.zeros((len(blk), 6))
            for _ in range(m):
                U = S.draw_latents(rng, len(blk))
                X = S.simulate_prevalence(blk, U)
                d = np.arange(len(blk))
                acc += std_summaries(X[d, d], mu, sd)
            s[a:a + len(blk)] = acc / m
        return np.linalg.norm(s - s_obs, axis=-1)

    def in_prior(th):
        b, e = th[:, 0], th[:, 1]
        return (e > 0) & (e < 0.5) & (b > e) & (b < 0.5)

    def log_prior(th):
        """log prior density, up to a constant, on the triangle.

        The prior eta ~ U(0,0.5), beta|eta ~ U(eta,0.5) has
        density 2/(0.5 - eta).
        """
        if prior == "published":
            return -np.log(np.maximum(0.5 - th[:, 1], 1e-12))
        return np.zeros(len(th))

    theta = (S.sample_theta_prior(rng, n_pop) if prior == "published"
             else S.sample_theta_prior_uniform(rng, n_pop))
    d = dist(theta)
    w = np.full(n_pop, 1.0 / n_pop)

    for _ in range(max_gen):
        if budget is not None and calls["n"] + n_pop * m > budget:
            stop_reason = "budget"
            break
        eps = float(np.quantile(d, quantile))
        ess = float(1.0 / np.sum(w ** 2))
        n_uniq = len(np.unique(np.round(theta, 12), axis=0))
        if trace is not None:
            trace.append({"calls": calls["n"], "eps": eps, "ess": ess,
                          "n_unique": n_uniq,
                          "mean": (theta * w[:, None]).sum(0).tolist()})
        if n_uniq < min_unique_frac * n_pop:
            stop_reason = "population_collapse"
            break
        stalled = stalled + 1 if eps > (1.0 - tol_rel) * eps_prev else 0
        eps_prev = min(eps, eps_prev)
        if stalled >= patience:
            stop_reason = "tolerance_plateau"
            break
        keep = d <= np.quantile(d, quantile)
        th_k, w_k = theta[keep], w[keep]
        w_k = w_k / w_k.sum()
        cov = 2.0 * np.cov(th_k.T, aweights=w_k) + 1e-12 * np.eye(2)
        L = np.linalg.cholesky(cov)
        idx = rng.choice(len(th_k), n_pop, p=w_k)
        prop = th_k[idx] + rng.standard_normal((n_pop, 2)) @ L.T
        bad = ~in_prior(prop)
        tries = 0
        while bad.any() and tries < 100:
            j = rng.choice(len(th_k), int(bad.sum()), p=w_k)
            prop[bad] = th_k[j] + rng.standard_normal((int(bad.sum()), 2)) @ L.T
            bad = ~in_prior(prop)
            tries += 1
        prop[bad] = th_k[rng.choice(len(th_k), int(bad.sum()), p=w_k)]
        theta = prop
        d = dist(theta)
        # uniform prior over the triangle / kernel-mixture proposal
        diff = theta[:, None, :] - th_k[None, :, :]
        sol = np.linalg.solve(cov, diff.reshape(-1, 2).T).T.reshape(
            len(theta), len(th_k), 2)
        lq = -0.5 * np.einsum("nkd,nkd->nk", diff, sol)
        lq = np.log((np.exp(lq - lq.max(1, keepdims=True)) * w_k).sum(1)
                    + 1e-300) + lq.max(1)
        lw = log_prior(theta) - lq
        w = np.exp(lw - lw.max())
        w /= w.sum()

    keep = d <= np.quantile(d, quantile)
    if trace is not None:
        trace.append({"stop_reason": stop_reason, "calls": calls["n"]})
    return theta[keep], w[keep] / w[keep].sum(), calls["n"]


# ----------------------------------------------------------------------
# 3. adSGLD proxy: kernel-score SGLD with finite-difference gradients
# ----------------------------------------------------------------------
def adsgld(s_obs, mu, sd, gamma, rng, J=8, beta_temp=10.0, dt=5e-2,
           n_steps=1200, burn_in=600, thin=2, h=1e-2, prior="published"):
    """
    SGLD on the same MMD-Bayes target. We don't compare with adSGLD, since the adaptive 
    preconditioner's second-moment estimate is dominated by finite-difference noise here.

    beta_temp TRACKS df_etld's. Under the sum-over-observations
    convention beta = w * n, and this benchmark has a single observed
    series (n = 1), so beta = w and the two methods must carry the same
    number to target the same posterior. df_etld's was calibrated to
    nominal coverage (30 -> 10); this follows it, which is what keeps
    DF-ETLD vs adSGLD a comparison of SAMPLERS rather than of two
    different posteriors.

    Step size: dt = 5e-2, chosen by a sweep (5e-3 ... 5e-2). At small dt the
    chain stalls in the saturated corner of the logistic reparameterisation,
    where the simulator is nearly flat in z and the FD gradient vanishes, so
    only the weak prior drift remains; it then reports the prior corner as
    its posterior. DF-ETLD does not have this problem because the
    ensemble covariance C_s preconditions the dynamics automatically -- the
    property adSGLD's adaptive preconditioner has to supply by hand, and
    which this FD proxy does not reproduce.
    """
    calls = {"n": 0}

    def mmd2(z, U):
        calls["n"] += U.shape[0]
        X = S.simulate_prevalence(S.z_to_theta(z[None]), U)[0]
        x = std_summaries(X, mu, sd)                      # (J, 6)
        d2 = ((x[:, None] - x[None, :]) ** 2).sum(-1)
        K = np.exp(-d2 / (2 * gamma ** 2))
        np.fill_diagonal(K, 0.0)
        Jn = x.shape[0]
        t_xx = K.sum() / (Jn * (Jn - 1))
        d2y = ((x - s_obs[None]) ** 2).sum(-1)
        t_xy = np.exp(-d2y / (2 * gamma ** 2)).mean()
        return t_xx - 2.0 * t_xy

    grad_lp = (S.grad_log_prior_z if prior == "published"
               else S.grad_log_prior_z_uniform)
    z = (S.sample_z_prior(rng, 1)[0] if prior == "published"
         else S.sample_z_prior_uniform(rng, 1)[0])
    out = []
    for t in range(n_steps):
        U = S.draw_latents(rng, J)                        # common randoms
        g = np.zeros(2)
        for d in range(2):
            e = np.zeros(2); e[d] = h
            g[d] = (mmd2(z + e, U) - mmd2(z - e, U)) / (2 * h)
        grad = np.ravel(grad_lp(z[None])) - beta_temp * g
        grad = grad / (1.0 + dt * np.abs(grad))           # tamed, as ours
        z = z + dt * grad + np.sqrt(2 * dt) * rng.standard_normal(2)
        if t >= burn_in and (t - burn_in) % thin == 0:
            out.append(z.copy())
    return S.z_to_theta(np.array(out)), calls["n"]


# ----------------------------------------------------------------------
# 3b. SGLD on the ENERGY score
# ----------------------------------------------------------------------
def adsgld_energy(s_obs, mu, sd, rng, J=8, beta_temp=10.0, dt=5e-2,
                  n_steps=1200, burn_in=600, thin=2, h=1e-2,
                  prior="published"):
    """
    The same sampler as `sgld`, on the ENERGY score instead of the
    kernel score. Everything else is held fixed -- z-space, finite
    difference gradients, step size, beta, the prior, and the common
    random numbers -- so the pair isolates the SCORING RULE, exactly as
    exp2_contamination does for the contamination study.

        ES(P, y) = E||X - y|| - 1/2 E||X - X'||

    estimated from the J simulated summary vectors,

        ES_hat = (1/J) sum_j ||x_j - y||
                 - 1/(2 J (J-1)) sum_{j != l} ||x_j - x_l||

    negatively oriented, so the generalized posterior is
    pi(theta) prop. prior * exp(-beta * ES_hat) -- the same sign
    convention the kernel-score arm uses for MMD^2.

    NO BANDWIDTH. The energy score is scale-free, so unlike the kernel
    arm there is no gamma to choose and none to get wrong. Summaries are
    still standardised by the frozen prior-predictive mu/sd, because the
    six summaries have wildly different units and an unstandardised
    Euclidean norm would be dominated by whichever has the largest scale.
    """
    calls = {"n": 0}

    def energy(z, U):
        calls["n"] += U.shape[0]
        X = S.simulate_prevalence(S.z_to_theta(z[None]), U)[0]
        x = std_summaries(X, mu, sd)                      # (J, 6)
        Jn = x.shape[0]
        d_xy = np.linalg.norm(x - s_obs[None], axis=-1).mean()
        d_xx = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
        # U-statistic over the J(J-1) ordered off-diagonal pairs
        t_xx = d_xx.sum() / (Jn * (Jn - 1))
        return d_xy - 0.5 * t_xx

    grad_lp = (S.grad_log_prior_z if prior == "published"
               else S.grad_log_prior_z_uniform)
    z = (S.sample_z_prior(rng, 1)[0] if prior == "published"
         else S.sample_z_prior_uniform(rng, 1)[0])
    out = []
    for t in range(n_steps):
        U = S.draw_latents(rng, J)                        # common randoms
        g = np.zeros(2)
        for d in range(2):
            e = np.zeros(2); e[d] = h
            g[d] = (energy(z + e, U) - energy(z - e, U)) / (2 * h)
        grad = np.ravel(grad_lp(z[None])) - beta_temp * g
        grad = grad / (1.0 + dt * np.abs(grad))           # tamed, as ours
        z = z + dt * grad + np.sqrt(2 * dt) * rng.standard_normal(2)
        if t >= burn_in and (t - burn_in) % thin == 0:
            out.append(z.copy())
    return S.z_to_theta(np.array(out)), calls["n"]


# ----------------------------------------------------------------------
# posterior-predictive summary check (the method-agnostic analogue of
# Kelly et al.)
# ----------------------------------------------------------------------
def predictive_summaries(theta_draws, mu, sd, rng, n_pred=200,
                         weights=None, chunk=25):
    """
    Simulate at posterior draws and return standardised summaries
    (n_pred, 6). Comparing the observed standardised summary against this
    distribution shows WHICH summary the fitted model cannot reproduce,
    computable for any posterior sampler.

    TWO THINGS THIS HAS TO GET RIGHT, both of which are easy to get wrong
    and both of which change what the figure says:

    INDEPENDENT NOISE PER DRAW. simulate_prevalence broadcasts U across
    every particle, so passing draw_latents(rng, 1) would give all n_pred
    draws the SAME Brownian path -- a predictive that integrates over the
    posterior but NOT over the simulator noise. In this model the OU noise
    (sigma = 0.5) is most of the variability: measured, that collapses the
    predictive sd of max-day, half-day and autocorrelation by 2.1-2.5x,
    which in turn reported the lag-1 autocorrelation -- the summary this
    benchmark exists to flag -- at z = 0.04 (perfectly compatible) instead
    of -1.56. Each draw therefore gets its own path, in chunks to keep the
    simulator vectorised.

    WEIGHTS. SMC-ABC returns a WEIGHTED particle cloud (measured ESS 16 of
    50 particles here), so resampling it uniformly would show a predictive
    for a posterior no method actually reports. figure7 already uses the
    weights; pass them here too.
    """
    p = None if weights is None else np.asarray(weights) / np.sum(weights)
    idx = rng.choice(len(theta_draws), n_pred, p=p)
    out = np.empty((n_pred, 6))
    for a in range(0, n_pred, chunk):
        blk = idx[a:a + chunk]
        # one path per draw: simulate the block against a block of paths
        # and read the diagonal, so the vectorisation is kept without
        # sharing noise across draws
        U = S.draw_latents(rng, len(blk))
        X = S.simulate_prevalence(theta_draws[blk], U)      # (b, b, T)
        d = np.arange(len(blk))
        out[a:a + len(blk)] = std_summaries(X[d, d], mu, sd)
    return out
