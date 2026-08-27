"""
experiment.py
=============
Orchestration: run one configuration over R seeds, and run the sweeps that
produce the two paper figures (A: bias vs nonlinearity; B: bias vs
ensemble size). Results are plain dicts -> JSON-serializable, so sweeps
can be cached to disk and re-plotted without re-running.

Seeding protocol (important for a clean bias attribution):
  * Seed r uses rng = default_rng(cfg.seed0 + r).
  * Within a seed, ONE dataset y is drawn first, and the SAME y, the SAME
    bandwidth gamma (median heuristic on y), and the SAME reference posterior
    are shared by both samplers. Each sampler then gets its own child RNG
    spawned from the seed, so sampler-internal randomness (init, Brownian
    increments, latent draws u^j) is independent across methods but the
    statistical problem they face is identical.
  * gamma is computed once per dataset and frozen (no per-method adaptation).
"""

import json
import numpy as np

from config import Config, with_
from model import generate_data, median_heuristic
from reference import ReferencePosterior
from samplers import aldi_exact_gradient, df_etld
from metrics import w1_to_reference, moment_errors


def run_single_seed(cfg: Config, seed_offset: int) -> dict:
    """One dataset -> reference + both samplers -> metrics."""
    root = np.random.default_rng(cfg.seed0 + seed_offset)
    rng_data, rng_aldi, rng_df = root.spawn(3)

    # ---- shared statistical problem ----
    y = generate_data(cfg.theta_star, cfg.alpha, cfg.sigma, cfg.n_obs,
                      rng_data)
    gamma = median_heuristic(y)
    ref = ReferencePosterior(y, cfg.alpha, cfg.sigma, gamma, cfg.beta,
                             cfg.m0, cfg.s0,
                             cfg.grid_lo, cfg.grid_hi, cfg.grid_size)

    # ---- initialization ----------
    # (inverse-CDF sampling from the grid posterior). Both samplers get
    # their own independent draw. This isolates stationary bias: correct
    # invariant measure -> stays at the finite-sample floor; biased ->
    # drifts to its own stationary state = the quantity we measure.
    def make_init(rng):
        if cfg.init_mode == "reference":
            return ref.quantile(rng.uniform(size=cfg.M))
        return None                          # -> prior init inside sampler

    # ---- rung 2: exact gradient ----
    s_aldi, d_aldi = aldi_exact_gradient(cfg, y, gamma, rng_aldi,
                                         theta0=make_init(rng_aldi))
    # ---- rung 3: DF-ETLD ----
    s_df, d_df = df_etld(cfg, y, gamma, rng_df, theta0=make_init(rng_df))

    def pack(samples, diag, extra=()):
        w1 = w1_to_reference(samples, ref)
        d = {"w1": w1,
             # RELATIVE W1: raw W1 is not comparable across alpha because
             # the posterior narrows as g'(theta*) grows; dividing by the
             # reference std makes the y-axis mean the same thing at every
             # alpha ("error in units of posterior spread").
             "w1_rel": w1 / ref.std,
             **moment_errors(samples, ref),
             "min_C_s": diag["min_C_s"],
             "tame_stationary": diag["tame_stationary"]}
        for k in extra:
            d[k] = diag[k]
        return d

    return {"gamma": gamma,
            "aldi": pack(s_aldi, d_aldi),
            "dfetld": pack(s_df, d_df, extra=("mean_C_stationary",))}


def run_config(cfg: Config, n_seeds=None):
    """
    Run cfg over n_seeds repetitions; aggregate mean +/- std of each metric.
    """
    n_seeds = cfg.n_seeds if n_seeds is None else n_seeds
    per_seed = [run_single_seed(cfg, r) for r in range(n_seeds)]

    def agg(method, key):
        # std: seed-to-seed spread (sample std, ddof=1 -- unbiased, which
        #   matters at the small R of the demo runs).
        # sem: uncertainty OF THE MEAN, std/sqrt(R). Figures plot means, so
        #   sem is the correct bar; std is kept for the "how variable is a
        #   single run" question and for the unit-test tolerance.
        vals = np.array([p[method][key] for p in per_seed])
        R = len(vals)
        sd = float(vals.std(ddof=1)) if R > 1 else 0.0
        return {"mean": float(vals.mean()), "std": sd,
                "sem": sd / np.sqrt(R)}

    result = {"n_seeds": n_seeds}
    for method in ("aldi", "dfetld"):
        result[method] = {k: agg(method, k)
                          for k in ("w1", "w1_rel", "mean_err", "std_err",
                                    "tame_stationary")}
    result["dfetld"]["mean_C_stationary"] = agg("dfetld",
                                                "mean_C_stationary")
    return result


# ----------------------------------------------------------------------
# sweeps for the two figures
# ----------------------------------------------------------------------
def sweep_alpha(base: Config, alphas, n_seeds=None):
    """Figure A: W1 vs alpha for both rungs."""
    return {a: run_config(with_(base, alpha=a), n_seeds) for a in alphas}


def sweep_M(base: Config, Ms, alphas, n_seeds=None):
    """Figure B: W1 vs ensemble size M, one curve per alpha."""
    return {a: {M: run_config(with_(base, alpha=a, M=M), n_seeds)
                for M in Ms} for a in alphas}


def save_json(obj, path):
    """Cache sweep results (numpy arrays are stripped by the agg step)."""
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()
                    if not str(k).startswith("_")}
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    with open(path, "w") as f:
        json.dump(clean(obj), f, indent=1)
