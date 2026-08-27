"""
exp3_sir/run_sir_es.py
======================
The ENERGY-score SGLD arm, run to convergence, and the paper's SIR
posterior figure.

Only the SGLD arm is run here. DF-ETLD's and SMC-ABC's draws are read
back from outputs/posteriors.npz (written by run_sir.py), so all three
panels describe the SAME observed series and the same standardisation --
the figure differs from sir_posteriors.png in exactly one thing, the
scoring rule the gradient-based sampler is given.

CONVERGENCE. The paper's configuration is 4 independent chains x 16 000
steps (thin 1, J = 24, burn-in one half), pooled to 32 000 draws, with
split-Rhat computed over the pooled half-chains and reported as measured.
This is the run behind the adSGLD-ES row of the paper's SIR table and the
right-hand panel of the SIR posterior figure.

    python run_sir_es.py                          # paper scale (4 x 16k)
    python run_sir_es.py --chains 2 --steps 400   # smoke run

Writes alongside the main outputs rather than over them:
sir_posteriors_es.png, sir_posteriors_es_credible.png, posteriors_es.npz,
results_es.json.
"""
import argparse
import json
import os
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np

import sir_model as S
import methods as M
import plots as P

ORDER_ES = ["dfetld", "abc", "adsgld_es"]


def _run_chain(seed, s_obs, mu, sd, beta, J, n_steps, thin):
    """One independent adSGLD-ES chain; returns (draws, sim_calls)."""
    rng = np.random.default_rng(seed)
    return M.adsgld_energy(s_obs, mu, sd, rng, J=J, beta_temp=beta,
                           n_steps=n_steps, burn_in=n_steps // 2,
                           thin=thin)


def split_rhat(chains):
    """
    Split-Rhat per parameter (Gelman et al.), on the post-burn-in draws:
    each chain is split in half, giving 2 x n_chains sequences.
    chains: list of (n_kept, D) arrays with equal n_kept.
    """
    halves = []
    for c in chains:
        n = len(c) // 2
        halves += [c[:n], c[n:2 * n]]
    x = np.stack(halves)                     # (m, n, D)
    m, n, _ = x.shape
    mean_j = x.mean(1)                       # (m, D)
    B = n * mean_j.var(0, ddof=1)
    W = x.var(1, ddof=1).mean(0)
    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--J", type=int, default=24)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--thin", type=int, default=1)
    # beta from calibrate_beta_es.py (Pacchiardi et al.'s
    # Bayes-factor matching); None -> read the saved value
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    d = np.load(os.path.join(args.outdir, "posteriors.npz"))
    s_obs = d["s_obs"]
    print(f"loaded DF-ETLD ({len(d['dfetld'])} draws) and "
          f"SMC-ABC ({len(d['abc'])} draws) from {args.outdir}/")

    # the standardiser is rebuilt from a fixed seed so mu/sd are the same
    # on every invocation -- they are not persisted in the npz
    rng = np.random.default_rng(args.seed)
    mu, sd, gamma, _ = M.build_standardiser(rng)
    print(f"standardiser rebuilt (seed {args.seed}); gamma {gamma:.3f} "
          f"(unused by the energy score)")

    beta = args.beta
    if beta is None:
        cal = json.load(open(os.path.join(args.outdir,
                                          "beta_es_calibration.json")))
        beta = cal["beta_es"]
        print(f"beta_ES = {beta:.4f} (calibrated: {cal['method']}, "
              f"anchored on beta_KS = {cal['beta_ref_ks']:g})")

    seeds = [args.seed + 1000 * (c + 1) for c in range(args.chains)]
    worker = partial(_run_chain, s_obs=s_obs, mu=mu, sd=sd, beta=beta,
                     J=args.J, n_steps=args.steps, thin=args.thin)
    t = time.perf_counter()
    with Pool(min(args.chains, cpu_count())) as pool:
        results = pool.map(worker, seeds)
    wt = time.perf_counter() - t
    chains = [r[0] for r in results]
    calls = int(sum(r[1] for r in results))
    th_es = np.concatenate(chains)
    rhat = split_rhat(chains)
    print(f"adSGLD-ES  {wt:6.1f}s  mean(beta,eta)="
          f"{np.round(th_es.mean(0), 4)}  calls={calls}  "
          f"split-Rhat={np.round(rhat, 4)}")
    print(f"truth                 ={np.round(S.THETA_TRUE, 4)}")

    post = {"dfetld": (d["dfetld"], None),
            "abc": (d["abc"], d["abc_w"]),
            "adsgld_es": (th_es, None)}
    P.sir_posteriors(post, os.path.join(args.outdir, "sir_posteriors_es.png"),
                     order=ORDER_ES)
    P.sir_posteriors(post,
                     os.path.join(args.outdir,
                                  "sir_posteriors_es_credible.png"),
                     credible=[0.5, 0.8, 0.95], order=ORDER_ES)

    np.savez_compressed(os.path.join(args.outdir, "posteriors_es.npz"),
                        adsgld_es=th_es, s_obs=s_obs)
    err = float(np.linalg.norm(th_es.mean(0) - S.THETA_TRUE))
    json.dump({"post_mean_adsgld_es": th_es.mean(0).tolist(),
               "post_sd_adsgld_es": th_es.std(0).tolist(),
               "truth": S.THETA_TRUE.tolist(),
               "err_norm": err,
               "beta": beta,
               "beta_method": "Pacchiardi et al. Sec 4.2 "
                              "(Bayes-factor matching)",
               "sim_calls": calls, "wallclock_s": round(wt, 1),
               "n_chains": args.chains, "n_steps": args.steps,
               "J": args.J,
               "split_rhat": [round(float(r), 4) for r in rhat],
               "draws": int(len(th_es))},
              open(os.path.join(args.outdir, "results_es.json"), "w"),
              indent=1)
    print("wrote sir_posteriors_es.png, sir_posteriors_es_credible.png")


if __name__ == "__main__":
    main()
