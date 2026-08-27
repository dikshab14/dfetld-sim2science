"""
run_experiment.py
=================
Entry point.

    python run_experiment.py --demo    # same config, 4 seeds and a 40-time-unit
                                       # sampling window; minutes for A-B
    python run_experiment.py --full    # the sweeps behind Figures A-B
                                       # (~1.5 h serial at 10 seeds)

Both modes run the same dynamics on the same sweep grids; --demo only lowers
the seed count and the sampling length, so its numbers are comparable to the
figures, just noisier. See the --demo branch in main() for the exact deltas.

Both run the built-in unit test of the design first:

  UNIT TEST (alpha = 0): the forward map is affine, statistical linearization
  is exact (paper Proposition 1), so
  DF-ETLD and ALDI-exact-gradient must agree with the grid reference up to
  finite-M / discretization noise. If W1(DF-ETLD) is not comparable to
  W1(ALDI) at alpha = 0, something is wrong with the implementation, NOT
  with the method -- fix before trusting any other number.

Map choice: the default mean map is the monotone cubic g = theta + alpha
theta^3 (see model.MAP_KIND), which guarantees a unimodal reference for all
alpha, so W1 measures linearization bias and nothing else. The quadratic
variant is non-injective (fold at -1/(2 alpha)); validation runs showed its
mirror mode sits inside the prior already at alpha ~ 0.25, making the
reference bimodal -- keep the quadratic for the multimodality/tempering
section of the paper, not for this study.
"""

import argparse
import os
import time

import config as C
from config import Config, with_
from experiment import run_config, sweep_alpha, sweep_M, save_json
import plots


def unit_test_alpha0(cfg, n_seeds):
    print("=" * 60)
    print("UNIT TEST at alpha = 0 (linear map: linearization exact)")
    r = run_config(with_(cfg, alpha=0.0), n_seeds=n_seeds)
    w_a = r["aldi"]["w1_rel"]
    w_g = r["dfetld"]["w1_rel"]
    print(f"  relW1(ALDI exact grad) = {w_a['mean']:.4f} +/- {w_a['std']:.4f}")
    print(f"  relW1(DF-ETLD)         = {w_g['mean']:.4f} +/- {w_g['std']:.4f}")
    # Pass criterion: DF-ETLD within a few pooled stds of ALDI. Loose on
    # purpose -- this is a smoke alarm, not a hypothesis test.
    tol = w_a["mean"] + 4.0 * (w_a["std"] + w_g["std"]) + 0.02
    status = "PASS" if w_g["mean"] <= tol else "FAIL"
    print(f"  -> {status} (tolerance {tol:.4f})")
    print("=" * 60)
    return status == "PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--only", default="AB",
                    help="subset of figures to run, e.g. --only A or --only B"
                         " (unit test always runs first)")
    ap.add_argument("--beta", type=float, default=None,
                    help="override cfg.beta (posterior concentration; the "
                         "paper figures use the default beta = 100)")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.demo:
        # Demo is the PRODUCTION configuration with less precision -- not a
        # different experiment. Everything that can shift a result is inherited
        # from Config() unchanged: dt, the spin-up length burn_in, thin, and M.
        # Demo numbers are therefore directly comparable to the figures and
        # merely noisier. Only the two precision knobs are reduced:
        #   n_steps 30400 -> 4400: 40 time units of sampling instead of 300,
        #     i.e. ~20 tau rather than ~150 tau (tau ~ 2 time units, measured),
        #     so the effective sample size falls ~7x and W1 rises toward its
        #     Monte Carlo floor.
        #   n_seeds 10 -> 4: error bars ~1.6x wider, since SEM = std/sqrt(R).
        # The previous demo also overrode burn_in and thin, with values tuned
        # for dt = 1e-3. At dt = 1e-2 they left under 4 tau of sampling and
        # thinned 50x finer than tau, so demo output was pure noise floor and
        # not comparable to anything.
        base = Config(n_steps=4400, n_seeds=4)
        n_seeds = 4
    else:
        base = Config()
        n_seeds = base.n_seeds
    # Sweep grids come from config.py in BOTH modes, so --demo and --full cover
    # the same points and differ only in precision.
    alphas = C.ALPHA_SWEEP
    Ms = C.M_SWEEP
    alphas_M = C.ALPHA_FOR_M_SWEEP

    if args.beta is not None:
        base = with_(base, beta=args.beta)

    t0 = time.time()
    ok = unit_test_alpha0(base, n_seeds)
    if not ok:
        print("Unit test failed; aborting sweeps.")
        return

    if "A" in args.only:
        run_A(base, alphas, n_seeds, args.outdir)
    if "B" in args.only:
        print("\n[Figure B] M sweep:", Ms, "at alpha in", alphas_M)
        resB = sweep_M(base, Ms, alphas_M, n_seeds)
        plots.figure_B(resB, os.path.join(args.outdir, "bias_vs_ensemble_size.png"))
        save_json(resB, os.path.join(args.outdir, "bias_vs_ensemble_size.json"))

    print(f"\nDone in {time.time() - t0:.1f}s. Figures in {args.outdir}/")


def run_A(base, alphas, n_seeds, outdir):
    print("\n[Figure A] alpha sweep:", alphas)
    resA = sweep_alpha(base, alphas, n_seeds)
    for a in alphas:
        # printed +/- is the SEM, i.e. what Figure A now draws
        ra, rg = resA[a]["aldi"]["w1"], resA[a]["dfetld"]["w1"]
        tm = resA[a]["dfetld"]["tame_stationary"]["mean"]
        print(f"  alpha={a:<5} W1 ALDI {ra['mean']:.4f}+/-{ra['sem']:.4f}"
              f"   DF-ETLD {rg['mean']:.4f}+/-{rg['sem']:.4f}"
              f"   [stationary max dt|drift| ~ {tm:.3f}]")
    plots.figure_A(resA, os.path.join(outdir, "bias_vs_nonlinearity.png"))
    # the same bias in posterior-sd units: whether it MATTERS, as opposed
    # to how big it is in the parameter's own units
    plots.figure_A(resA, os.path.join(outdir,
                                      "bias_vs_nonlinearity_normalized.png"),
                   key="w1_rel")
    save_json(resA, os.path.join(outdir, "bias_vs_nonlinearity.json"))


if __name__ == "__main__":
    main()
