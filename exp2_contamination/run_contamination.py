"""
exp2_contamination/run_contamination.py
=======================================
Robustness to Huber eps-contamination, reported as posterior-mean error
against the contamination fraction:

    y_i ~ (1 - eps) N(theta*, 1) + eps N(z, 1),   theta* = 1, z = 10

ALIGNMENT. Every setting -- theta*, z, N, prior, data generation, bandwidth, sampler configuration
and above all the LEARNING RATES -- is imported from run_posterior_grid, and w
is READ FROM outputs/contamination_posterior_grid_config.json rather than re-fitted. exp3's
three figures are therefore three views of ONE experiment:

    contamination_posterior_grid.png   posterior shapes at 3 eps / 3 z values
    eps_sweep_breakdown.png     the whole admissible path, eps in [0, 0.5]
    contamination.png           this file: mean error vs eps, 4 methods

Why it matters that w is shared and not re-fitted: the generalized
posteriors are different objects whose raw concentrations are not
comparable, so a method can look robust merely by being wide. Each w was
calibrated ONCE on clean data to match standard Bayes' posterior sd, then
frozen. Differences between curves are then robustness, not width. The
posterior sd is logged per eps so that assumption stays checkable.

THE FOUR METHODS, and why this particular four:

  * standard Bayes -- conjugate, closed form, so it carries NO sampler
    error and its curve IS the contamination bias;
  * DF-ETLD -- ours: kernel score, derivative-free;
  * adSGLD on the KERNEL score -- the same posterior as DF-ETLD (same w,
    same gamma) reached with gradients. DF-ETLD against this isolates the
    SAMPLER;
  * adSGLD on the ENERGY score -- the same sampler as the above with a
    different scoring rule. Those two against each other isolate the
    SCORING RULE.

So the figure separates "is the derivative-free sampler paying a price"
from "is the kernel score buying robustness" -- which a three-method
version cannot do.

Both adSGLD gradients are EXACT, not estimated: for the location model
x = theta + sigma u we have dx/dtheta = 1, and both scores' model-model
terms are theta-independent (neither E|X - X'| nor E k(X, X') moves under
a location shift), so those gradients vanish and one pass over the data
suffices. The energy score additionally needs no bandwidth -- it is
scale-free.

    python run_contamination.py            # 10 seeds x 5 eps values
    python run_contamination.py --demo     # 3 seeds, short chains
"""

import argparse
import json
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_posterior_grid as R

EPS_GRID = [0.0, 0.1, 0.2, 0.3, 0.4]     # strictly inside the admissible
EPS_DEMO = [0.0, 0.2, 0.4]               # range; eps_sweep covers [0, 0.5]
Z_OUT = 10.0
N_SEEDS = 10
SEED0 = 3000


def adsgld_kernel_score(y, gamma, w, rng, n_steps=20000, J=50, dt=2e-4,
                        burn_frac=0.5, thin=5):
    """
    Plain SGLD on the KERNEL-score posterior -- the target DF-ETLD
    samples, at the same w and the same gamma, reached with gradients
    instead of statistical linearization.

    Reparameterize x^j = theta + sigma u^j so dx/dtheta = 1; then
    d/dtheta MMD^2_hat is the mean of the kernel's input-gradients (paper
    eq. (4) with dG/dtheta = 1), and beta = w * N reproduces df_etld's
    sum-over-observations convention exactly. Any gap between this curve
    and DF-ETLD's is therefore sampler, not target.

    UNPRECONDITIONED. With D = 1 an adaptive diagonal
    preconditioner has no anisotropy to correct, so it reduces to a step
    rescaling -- and RMSprop's rescaling normalizes the drift to
    ~step/2 whatever the gradient scale, which stops the chain reaching
    the mode and makes the w-calibration chase a non-stationary spread.
    The gradient here is exact and analytic, so there is no gradient
    noise for adaptivity to manage either. Hence SGLD, not adSGLD.
    """
    n = len(y)
    beta = w * n
    yv = y[:, 0]
    burn = int(burn_frac * n_steps)
    th = R.PRIOR_MEAN + R.PRIOR_STD * rng.standard_normal()
    out = []
    for t in range(n_steps):
        x = th + R.SIGMA * rng.standard_normal(J)
        dxx = x[:, None] - x[None, :]
        Kxx = np.exp(-dxx ** 2 / (2 * gamma ** 2))
        np.fill_diagonal(Kxx, 0.0)
        gm = (2.0 / (J * (J - 1))) * np.sum(-dxx / gamma ** 2 * Kxx)
        dxy = x[:, None] - yv[None, :]
        Kxy = np.exp(-dxy ** 2 / (2 * gamma ** 2))
        gd = (2.0 / (J * n)) * np.sum(-dxy / gamma ** 2 * Kxy)
        grad = -(th - R.PRIOR_MEAN) / R.PRIOR_STD ** 2 - beta * (gm - gd)
        th = th + dt * grad + np.sqrt(2 * dt) * rng.standard_normal()
        if t >= burn and (t - burn) % thin == 0:
            out.append(th)
    return np.array(out)


KEYS = ("bayes", "dfetld", "adsgld", "adsgld_es")


def run_seed(task, w_df, w_ad, cfg):
    """One (eps, seed): all four posteriors on the SAME dataset, so the
    comparison is paired and dataset variability cancels."""
    i, eps, seed = task
    rng = np.random.default_rng(SEED0 + 1000 * seed + i)
    y = R.generate_data(eps, Z_OUT, rng)
    g = R.model_gamma(rng)
    draws = {
        "bayes": R.standard_bayes(y, rng),
        "dfetld": R.df_etld(y, g, w_df, rng, M=cfg["M"], J=cfg["J"],
                            n_steps=cfg["n_steps"]),
        "adsgld": adsgld_kernel_score(y, g, w_df, rng,
                                      n_steps=cfg["ad_steps"]),
        "adsgld_es": R.adsgld_energy(y, w_ad, rng, n_steps=cfg["ad_steps"]),
    }
    return i, g, {k: (abs(float(v.mean()) - R.THETA_STAR), float(v.std()))
                  for k, v in draws.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cfg = ({"M": 25, "J": 10, "n_steps": 800, "ad_steps": 5000}
           if args.demo else
           {"M": 50, "J": 20, "n_steps": 2000, "ad_steps": 20000})
    eps_grid = EPS_DEMO if args.demo else EPS_GRID
    n_seeds = 3 if args.demo else N_SEEDS

    cfg_path = os.path.join(args.outdir, "contamination_posterior_grid_config.json")
    with open(cfg_path) as fh:
        base = json.load(fh)
    w_df, w_ad = base["w_kernel"], base["w_energy"]
    print(f"w from {cfg_path}: kernel {w_df:.3f}  energy {w_ad:.3f}  "
          f"(clean-data target sd {base['sd_target']:.4f})", flush=True)

    tasks = [(i, e, s) for i, e in enumerate(eps_grid)
             for s in range(n_seeds)]
    with Pool(6) as pool:
        res = pool.map(partial(run_seed, w_df=w_df, w_ad=w_ad, cfg=cfg),
                       tasks)

    acc = {i: {k: [] for k in KEYS + ("gamma",)}
           for i in range(len(eps_grid))}
    for i, g, d in res:
        acc[i]["gamma"].append(g)
        for k in KEYS:
            acc[i][k].append(d[k])

    results = {}
    hdr = "  ".join(f"{k:>10}" for k in KEYS)
    print(f"\n{'eps':>5} {'gamma':>6} {'':>4} {hdr}")
    for i, e in enumerate(eps_grid):
        results[e] = {
            k: {"mean_err": float(np.mean([r[0] for r in acc[i][k]])),
                "err_std": float(np.std([r[0] for r in acc[i][k]])),
                "post_std": float(np.mean([r[1] for r in acc[i][k]]))}
            for k in KEYS}
        r = results[e]
        print(f"{e:>5.2f} {np.mean(acc[i]['gamma']):>6.2f} {'|err|':>4} "
              + "  ".join(f"{r[k]['mean_err']:>10.3f}" for k in KEYS))
        print(f"{'':>5} {'':>6} {'sd':>4} "
              + "  ".join(f"{r[k]['post_std']:>10.3f}" for k in KEYS),
              flush=True)
        json.dump(results, open(os.path.join(args.outdir,
                                             "contamination.json"), "w"),
                  indent=1, default=float)   # incremental: a long sweep
                                             # always leaves usable output

    path = os.path.join(args.outdir, "contamination.json")
    json.dump(results, open(path, "w"), indent=1, default=float)
    from plots import figure_contamination
    figure_contamination(json.load(open(path)),
                         os.path.join(args.outdir, "contamination.png"),
                         n_seeds=n_seeds)
    print("\nwrote", path, "and contamination.png")


if __name__ == "__main__":
    main()
