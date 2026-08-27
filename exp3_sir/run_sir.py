"""
exp3_sir/run_sir.py
=============================
Runs DF-ETLD, SMC-ABC and adSGLD on the misspecified SIR benchmark and produces:

  sir_posteriors.png   -- bivariate (eta, beta) contours with marginals,
                          one panel per method
  cost_table.md/.json  -- simulator calls and wall-clock per method

    python run_sir.py --demo     # quick
    python run_sir.py            # paper scale
"""

import argparse
import json
import os
import time

import numpy as np

import sir_model as S
import methods as M
import plots as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="outputs")
    # beta_temp for BOTH Langevin samplers. Default None keeps each
    # function's calibrated value (10). Set explicitly to produce the
    # sharper, under-covering variant for comparison -- with n = 1
    # observed series beta = w * n = w, so one number drives both.
    ap.add_argument("--beta-temp", type=float, default=None)
    # "published": eta ~ U(0,0.5), beta|eta ~ U(eta,0.5)  (Kelly et al.)
    # "uniform":   beta, gamma ~ U(0,0.5) with gamma > beta rejected,
    #              i.e. uniform on the triangle -- a different measure
    ap.add_argument("--prior", choices=["published", "uniform"],
                    default="published")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # CONVERGED configurations, not budget-matched ones. Each method gets
    # what it needs; simulator calls are reported, not equalised.
    #   DF-ETLD  stable from ~750 steps: recomputing the posterior from
    #            growing prefixes of a 6000-step run moves the mean by
    #            <= 0.11 sd and the sd not at all. 1500 keeps margin.
    #   adSGLD   NOT stationary at J=8 no matter the chain length. At J=24,
    #            the first and second halves of the
    #            chain agree to 0.01 sd. J is nearly free here -- the
    #            simulator's 3650-substep loop dominates, so J=24 costs
    #            the same wall-clock as J=8.
    #   SMC-ABC  m=1 leaves the distance with a noise floor the tolerance
    #            cannot beat (it plateaus at 3.07). m=20 lowers the floor
    #            to 1.51 and moves the posterior from (0.27, 0.21) to
    #            (0.17, 0.15) against a truth of (0.15, 0.10). It now
    #            stops on its own tolerance plateau.
    if args.demo:
        M_ens, J, n_steps, n_pop, n_pred = 25, 5, 400, 100, 150
        ad_steps, ad_J, abc_m = 400, 12, 5
    else:
        # M=50, J=20, with beta_temp=10 RE-CALIBRATED at this exact
        # configuration: on 12 well-specified datasets.
        # at M=40, J=8. beta = 10 is nominal; 30 under-covers.
        M_ens, J, n_steps, n_pop, n_pred = 50, 20, 1500, 200, 400
        ad_steps, ad_J, abc_m = 2000, 24, 20

    rng = np.random.default_rng(1000 + args.seed)
    mu, sd, gamma, n_pp = M.build_standardiser(rng, n_pp=200)
    y = S.make_observation(np.random.default_rng(args.seed),
                           misspecified=True)
    s_obs = M.std_summaries(y, mu, sd)
    print(f"bandwidth {gamma:.3f}; observed raw autocorr "
          f"{S.summaries(y)[5]:.4f}  (paper: 0.9957)")

    post, pred, rows = {}, {}, []

    # ---- DF-ETLD --------------------------------------------------------
    t = time.perf_counter()
    bkw = {"prior": args.prior}
    if args.beta_temp is not None:
        bkw["beta_temp"] = args.beta_temp
    th, calls, diag = M.df_etld(s_obs, mu, sd, gamma, rng, M=M_ens, J=J,
                                n_steps=n_steps, burn_in=n_steps // 2, **bkw)
    wt = time.perf_counter() - t
    post["dfetld"] = (th, None)
    pred["dfetld"] = M.predictive_summaries(th, mu, sd, rng, n_pred)
    rows.append({"method": "DF-ETLD (ours)", "sim_calls": calls,
                 "wallclock_s": wt,
                 "notes": f"M={M_ens}, J={J}, {n_steps} steps; "
                          f"tame={diag['tame_stationary']:.2f}; "
                          f"no simulator derivatives"})
    print(f"DF-ETLD    {wt:6.1f}s  mean(beta,eta)={np.round(th.mean(0),4)} "
          f"calls={calls}")

    # ---- DF-ETLD at a small simulation budget ---------------------------
    t = time.perf_counter()
    th_b, calls_b, _ = M.df_etld(s_obs, mu, sd, gamma,
                                 np.random.default_rng(99), M=20, J=4,
                                 n_steps=125, burn_in=60, **bkw)
    wt_b = time.perf_counter() - t
    rows.append({"method": "DF-ETLD @ 10k-call budget", "sim_calls": calls_b,
                 "wallclock_s": wt_b,
                 "notes": "small-budget run: 10,000 simulator calls"})
    print(f"DF-ETLD@10k{wt_b:6.1f}s  mean={np.round(th_b.mean(0),4)} "
          f"calls={calls_b}")

    # ---- SMC-ABC --------------------------------------------------------
    t = time.perf_counter()
    # no budget: runs to its own tolerance plateau
    th_a, w_a, calls_a = M.smc_abc(s_obs, mu, sd, rng, n_pop=n_pop,
                                   m=abc_m, prior=args.prior)
    wt_a = time.perf_counter() - t
    post["abc"] = (th_a, w_a)
    # ABC returns a WEIGHTED cloud -- pass the weights, or the predictive
    # describes a posterior no method reports (ESS is far below n here)
    pred["abc"] = M.predictive_summaries(th_a, mu, sd, rng, n_pred,
                                         weights=w_a)
    rows.append({"method": "SMC-ABC (6 summaries)", "sim_calls": calls_a,
                 "wallclock_s": wt_a,
                 "notes": f"run to convergence (tolerance plateau); "
                          f"pop={n_pop}, m={abc_m}"})
    print(f"SMC-ABC    {wt_a:6.1f}s  mean="
          f"{np.round((th_a * w_a[:, None]).sum(0), 4)} calls={calls_a}")

    # ---- adSGLD ---------------------------------------------------------
    t = time.perf_counter()
    th_s, calls_s = M.adsgld(s_obs, mu, sd, gamma, rng, J=ad_J,
                             n_steps=ad_steps, burn_in=ad_steps // 2, **bkw)
    wt_s = time.perf_counter() - t
    post["adsgld"] = (th_s, None)
    pred["adsgld"] = M.predictive_summaries(th_s, mu, sd, rng, n_pred)
    rows.append({"method": "adSGLD (FD-gradient proxy)",
                 "sim_calls": calls_s, "wallclock_s": wt_s,
                 "notes": f"single chain, J={ad_J} (gradient noise is the "
                          f"binding constraint, not dt); FD gradients"})
    print(f"adSGLD     {wt_s:6.1f}s  mean={np.round(th_s.mean(0),4)} "
          f"calls={calls_s}")

    # ---- outputs ---------------------------------------------------------
    # one draw from the TRUE (misspecified) DGP
    P.sir_observed(y, os.path.join(args.outdir, "sir_observed_series.png"))
    P.sir_posteriors(post, os.path.join(args.outdir, "sir_posteriors.png"))
    md = P.cost_table(rows, os.path.join(args.outdir, "cost_table.md"),
                      os.path.join(args.outdir, "cost_table.json"))
    print("\n" + md)

    z = {n: float((s_obs[j] - pred["dfetld"][:, j].mean())
                  / max(pred["dfetld"][:, j].std(), 1e-9))
         for j, n in enumerate(S.SUMMARY_NAMES)}
    print("DF-ETLD posterior-predictive z per summary:",
          {k: round(v, 2) for k, v in z.items()})
    json.dump({"post_mean_dfetld": th.mean(0).tolist(),
               "post_mean_abc": (th_a * w_a[:, None]).sum(0).tolist(),
               "post_mean_adsgld": th_s.mean(0).tolist(),
               "truth": S.THETA_TRUE.tolist(),
               "obs_autocorr": float(S.summaries(y)[5]),
               "predictive_z": z},
              open(os.path.join(args.outdir, "results.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(args.outdir, "posteriors.npz"),
                        dfetld=th, abc=th_a, abc_w=w_a, adsgld=th_s,
                        s_obs=s_obs, y=y)


if __name__ == "__main__":
    main()
