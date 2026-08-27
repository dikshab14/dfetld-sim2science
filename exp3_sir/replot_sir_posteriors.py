"""
exp3_sir/replot_sir_posteriors.py
=================================
Re-render the SIR posterior figures from the COMMITTED posterior draws,
without re-running any inference.

    python replot_sir_posteriors.py            # -> outputs/sir_posteriors_es*.png

Reads outputs/posteriors.npz (DF-ETLD and SMC-ABC draws, written by
run_sir.py) and outputs/posteriors_es.npz (the converged SGLD-ES draws,
written by run_sir_es.py), and redraws the same figures run_sir_es.py
produces. Use this after a styling or labelling change; the sampler is
untouched, so the figures are identical up to the styling itself.
"""
import argparse
import os

import numpy as np

import plots as P

ORDER_ES = ["dfetld", "abc", "adsgld_es"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    d = np.load(os.path.join(args.outdir, "posteriors.npz"))
    de = np.load(os.path.join(args.outdir, "posteriors_es.npz"))
    post = {"dfetld": (d["dfetld"], None),
            "abc": (d["abc"], d["abc_w"]),
            "adsgld_es": (de["adsgld_es"], None)}
    print(f"loaded DF-ETLD ({len(d['dfetld'])}), SMC-ABC ({len(d['abc'])}), "
          f"SGLD-ES ({len(de['adsgld_es'])}) draws from {args.outdir}/")

    P.sir_posteriors(post, os.path.join(args.outdir, "sir_posteriors_es.png"),
                     order=ORDER_ES)
    P.sir_posteriors(post,
                     os.path.join(args.outdir,
                                  "sir_posteriors_es_credible.png"),
                     credible=[0.5, 0.8, 0.95], order=ORDER_ES)
    print("wrote sir_posteriors_es.png, sir_posteriors_es_credible.png")


if __name__ == "__main__":
    main()
