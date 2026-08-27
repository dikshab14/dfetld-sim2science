"""
exp2_contamination/plots.py
===========================
Figure for the contamination experiment: posterior-mean error vs
contamination fraction eps, four methods -- standard Bayes, DF-ETLD
(kernel score, derivative-free), and adSGLD on BOTH the kernel score and
the energy score. Read in pairs: DF-ETLD vs adSGLD-kernel isolates the
SAMPLER (identical target), adSGLD-kernel vs adSGLD-energy isolates the
SCORING RULE (identical sampler).

The reference line eps(z - theta*) is drawn because the exact conjugate
posterior mean is driven by the outliers at exactly that rate -- so a
standard-Bayes curve tracking it is not a sampler failure but the
likelihood posterior behaving as theory says it must. That contrast is
the point of the figure.

Colours and styling follow the other exp3 figures (standard Bayes blue,
DF-ETLD orange, energy-score adSGLD green); the kernel-score chain takes
the spare red so no two methods ever share a colour across exp3.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(
    _os.path.abspath(__file__)), '..'))
from common import style
style.apply()

Z_OUT = 10.0
THETA_STAR = 1.0

# DF-ETLD and adSGLD-kernel target the IDENTICAL posterior, so their
# curves coincide -- which is the point, but a solid line drawn over a
# solid line just hides one method. The kernel-score chain is therefore
# dashed with open markers, so agreement reads as agreement rather than
# as a missing series.
STYLES = [("bayes", style.color("bayes"), style.label("bayes"),
           "-", "o", True),
          ("dfetld", style.color("dfetld"), style.label("dfetld"),
           "-", "o", True),
          # both gradient-based arms here are PLAIN SGLD (see
          # run_contamination.adsgld_kernel_score): colours unchanged,
          # names without the "ad" prefix.
          ("adsgld", style.color("adsgld_ks"), style.label("sgld_ks"),
           "--", "s", False),
          ("adsgld_es", style.color("adsgld_es"), style.label("sgld_es"),
           "-", "^", True)]


def figure_contamination(results, path, n_seeds=None):
    eps = sorted(float(e) for e in results)
    first = results[str(eps[0])]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for key, c, lab, ls, mk, filled in STYLES:
        if key not in first:
            continue          # older result files predate the energy-score
                              # chain: plot what is there, do not crash
        m = [results[str(e)][key]["mean_err"] for e in eps]
        s = [results[str(e)][key]["err_std"] for e in eps]
        ax.errorbar(eps, m, yerr=s, marker=mk, ms=4.5, lw=1.6, ls=ls,
                    capsize=2.5, elinewidth=0.9, color=c, label=lab,
                    mfc=c if filled else "none")
    ax.plot(eps, [e * (Z_OUT - THETA_STAR) for e in eps], "k:", lw=1,
            label=r"$\epsilon\,(z-\theta^*)$ (contamination bias)")
    ax.set_xlabel(r"contamination fraction $\epsilon$")
    ax.set_ylabel(r"$|\mathbb{E}[\theta\,|\,Y] - \theta^*|$")
    ax.legend(frameon=False, fontsize=13)
    fig.tight_layout()
    style.save(fig, path)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "outputs/contamination.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs/contamination.png"
    figure_contamination(json.load(open(src)), out)
    print("wrote", out)
