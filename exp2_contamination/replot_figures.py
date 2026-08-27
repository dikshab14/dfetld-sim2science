"""
exp2_contamination/replot_figures.py
====================================
Re-render both contamination figures from the COMMITTED results, without
re-running any inference.

    python replot_figures.py     # -> outputs/contamination_posterior_grid.png
                                 #    outputs/contamination.png

Reads
  outputs/contamination_posterior_grid_samples.npz  (posterior draws per panel)
  outputs/contamination_posterior_grid_config.json  (n_seeds, calibrated w)
  outputs/contamination.json                        (mean error vs eps)

and redraws exactly what run_posterior_grid.py and run_contamination.py
produce. Use this after a styling or labelling change: the samplers are
untouched, so the figures differ only in the styling itself.
"""
import argparse
import json
import os

import numpy as np

import run_posterior_grid as R
import plots as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    # ---- main-text figure: the 3 x 3 posterior grid ----------------------
    npz = os.path.join(args.outdir,
                       "contamination_posterior_grid_samples.npz")
    cfg = json.load(open(os.path.join(
        args.outdir, "contamination_posterior_grid_config.json")))
    d = np.load(npz)
    out = {}
    for full in d.files:                      # keys are "<panel>__<method>"
        key, m = full.rsplit("__", 1)
        out.setdefault(key, {})[m] = (float(d[full][0]) if m == "gamma"
                                      else d[full])
    print(f"loaded {len(out)} panels from {os.path.basename(npz)} "
          f"({cfg['n_seeds']} seeds, w_kernel={cfg['w_kernel']:.3f}, "
          f"w_energy={cfg['w_energy']:.3f})")
    grid_path = os.path.join(args.outdir, "contamination_posterior_grid.png")
    R.make_figure(out, grid_path, cfg["n_seeds"])
    print(f"wrote {os.path.basename(grid_path)}")

    # ---- appendix figure: mean error vs eps, four methods ----------------
    res_path = os.path.join(args.outdir, "contamination.json")
    results = json.load(open(res_path))
    err_path = os.path.join(args.outdir, "contamination.png")
    P.figure_contamination(results, err_path, n_seeds=cfg["n_seeds"])
    print(f"wrote {os.path.basename(err_path)}")


if __name__ == "__main__":
    main()
