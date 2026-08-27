"""
exp3_sir/calibrate_beta_es.py
=============================
Choose the energy-score weight by Pacchiardi et al.'s own procedure
(Generalized Bayesian likelihood-free inference, Sec. 4.2),
which adapts Bissiri, Holmes & Walker (2016).

THE METHOD. For a scoring-rule posterior pi_S(theta|y) prop.
prior * exp(-w S(P_theta, y)), the Bayes factor between two parameter
values is

    log[ pi_S(theta|y)/pi_S(theta'|y) / (pi(theta)/pi(theta')) ]
        = -w { S(P_theta, y) - S(P_theta', y) }.

Two SR posteriors therefore have the same SCALE when their Bayes factors
agree. Taking S' as a reference with weight w', that means

    w { S(theta) - S(theta') } = w' { S'(theta) - S'(theta') },

and since no particular pair (theta, theta') is preferred, w is set to
w' times the MEDIAN of the ratio over pairs drawn from the prior. S and
S' are estimated by simulating m draws at each theta, so the procedure
needs no likelihood and -- crucially -- no repeated posterior inference.

    python calibrate_beta_es.py
"""
import argparse
import json
import os

import numpy as np

import sir_model as S
import methods as M


def kernel_score(x, y, gamma):
    """S_kernel(P_theta, y), the MMD^2 estimator the KS arm minimises:
    U-statistic model-model term minus twice the model-data term."""
    J = x.shape[0]
    d2 = ((x[:, None] - x[None, :]) ** 2).sum(-1)
    K = np.exp(-d2 / (2 * gamma ** 2))
    np.fill_diagonal(K, 0.0)
    t_xx = K.sum() / (J * (J - 1))
    t_xy = np.exp(-((x - y[None]) ** 2).sum(-1) / (2 * gamma ** 2)).mean()
    return t_xx - 2.0 * t_xy


def energy_score(x, y):
    """S_energy(P_theta, y) = E||X - y|| - 1/2 E||X - X'||."""
    J = x.shape[0]
    d_xy = np.linalg.norm(x - y[None], axis=-1).mean()
    d_xx = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    return d_xy - 0.5 * d_xx.sum() / (J * (J - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--m", type=int, default=24,
                    help="simulations per theta when estimating a score")
    ap.add_argument("--beta-ref", type=float, default=10.0,
                    help="the kernel-score arm's calibrated beta")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mu, sd, gamma, _ = M.build_standardiser(np.random.default_rng(0))
    d = np.load(os.path.join(args.outdir, "posteriors.npz"))
    y = d["s_obs"]
    print(f"gamma {gamma:.3f}; reference beta_KS = {args.beta_ref:g}")

    n = 2 * args.pairs
    th = S.sample_theta_prior(rng, n)
    ks = np.empty(n)
    es = np.empty(n)
    for i in range(n):
        U = S.draw_latents(rng, args.m)
        X = S.simulate_prevalence(th[i][None], U)[0]
        x = M.std_summaries(X, mu, sd)
        ks[i] = kernel_score(x, y, gamma)
        es[i] = energy_score(x, y)
    calls = n * args.m

    a, b = ks[:args.pairs], ks[args.pairs:]
    p, q = es[:args.pairs], es[args.pairs:]
    d_ks, d_es = a - b, p - q
    # drop pairs whose energy-score difference is numerically zero: the
    # ratio is undefined there, and Pacchiardi et al.'s median is over
    # well-defined ratios
    ok = np.abs(d_es) > 1e-12
    ratio = d_ks[ok] / d_es[ok]
    beta_es = args.beta_ref * float(np.median(ratio))

    print(f"\n{args.pairs} prior pairs, m = {args.m} sims each "
          f"({calls:,} simulator calls)")
    print(f"  |delta S_kernel| median = {np.median(np.abs(d_ks)):.4f}")
    print(f"  |delta S_energy| median = {np.median(np.abs(d_es)):.4f}")
    print(f"  median ratio dKS/dES    = {np.median(ratio):.4f}")
    print(f"  implied kernel-vs-energy scale factor = "
          f"{1/np.median(ratio):.1f}x   (theirs on g-and-k: 28.1)")
    print(f"\n  beta_ES = beta_KS x median ratio = {args.beta_ref:g} x "
          f"{np.median(ratio):.4f} = {beta_es:.4f}")
    lo, hi = np.percentile(ratio, [25, 75])
    print(f"  (IQR of the ratio: {args.beta_ref*lo:.4f} to "
          f"{args.beta_ref*hi:.4f})")

    json.dump({"beta_es": beta_es, "beta_ref_ks": args.beta_ref,
               "median_ratio": float(np.median(ratio)),
               "ratio_iqr": [float(lo), float(hi)],
               "n_pairs": args.pairs, "m": args.m, "gamma": float(gamma),
               "sim_calls": int(calls),
               "method": "Pacchiardi et al. Sec 4.2 (Bayes-factor matching)"},
              open(os.path.join(args.outdir, "beta_es_calibration.json"), "w"),
              indent=1)
    print(f"\nwrote {args.outdir}/beta_es_calibration.json")


if __name__ == "__main__":
    main()
