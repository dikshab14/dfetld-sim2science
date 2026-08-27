"""
plots.py
========
The two figures of the nonlinearity-bias study (the paper's appendix
figure panels):

  Figure A             : W1-to-reference vs alpha; ALDI-exact-gradient vs
                         DF-ETLD. The vertical gap IS the linearization bias.
  Figure B             : W1 vs ensemble size M (log-log), colored by alpha.

All plotting reads the plain dict output of experiment.py, so figures can be
re-styled without re-running any dynamics.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(
    _os.path.abspath(__file__)), '..'))
from common import style
style.apply()
from matplotlib import ticker


def _plain_log_axis(ax, which, values=None, subs=(1, 2, 5)):
    """
    Label a log axis with the numbers it actually represents.

    Matplotlib's default log formatter prints powers of ten ("10^1", "10^-2")
    plus a swarm of unlabelled minor ticks. For an ensemble-size axis that is
    actively unhelpful: it never shows WHICH M were run. Put ticks on the real
    values and print them as plain decimals.
    """
    axis = ax.xaxis if which == "x" else ax.yaxis
    if values is not None:
        axis.set_major_locator(ticker.FixedLocator(list(values)))
    else:
        axis.set_major_locator(ticker.LogLocator(base=10, subs=subs))
    axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_locator(ticker.NullLocator())
    axis.set_minor_formatter(ticker.NullFormatter())


def _errbar(ax, xs, res_list, method, key="w1", err="sem", **kw):
    """
    Plot seed-averaged `key` with error bars.

    err = "sem" (default): standard error of the mean, std/sqrt(R). This is
      the uncertainty of the quantity actually drawn (a MEAN over R seeds).
      err = "std" restores the seed-to-seed spread, which answers a different
      question ("how much does one run vary?") and is sqrt(R) times wider --
      that is why the original figure's bars swamped the curves.
    """
    means = [r[method][key]["mean"] for r in res_list]
    stds = []
    for r in res_list:
        d = r[method][key]
        if err == "std":
            stds.append(d["std"])
        else:
            R = max(int(r.get("n_seeds", 1)), 1)
            stds.append(d.get("sem", d["std"] / np.sqrt(R)))
    kw.setdefault("marker", "o")      # overridable: Figure B marks ALDI with "s"
    ax.errorbar(xs, means, yerr=stds, capsize=3, **kw)


def figure_A(alpha_results, path, err="sem", key="w1"):
    """
    Linearization bias vs nonlinearity.

    key="w1"      -- W1 to the exact posterior in the parameter's own
                     units. The absolute size of the error.
    key="w1_rel"  -- the same distance divided by the exact posterior's
                     standard deviation, i.e. W1 measured in posterior
                     widths. This is the quantity that says whether the
                     bias MATTERS: an absolute W1 of 0.01 is negligible
                     against a posterior of sd 0.3 and fatal against one
                     of sd 0.003. The two are plotted separately because
                     the posterior width itself changes with alpha, so
                     the curves are not rescalings of one another.
    """
    alphas = sorted(alpha_results)
    res = [alpha_results[a] for a in alphas]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    _errbar(ax, alphas, res, "aldi", key=key, err=err,
            label=style.label("aldi"), color=style.color("aldi"))
    _errbar(ax, alphas, res, "dfetld", key=key, err=err,
            label=style.label("dfetld"), color=style.color("dfetld"))
    ax.set_xlabel(r"nonlinearity $\alpha$")
    ax.set_ylabel(r"$W_1$ / posterior sd" if key == "w1_rel"
                  else r"$W_1$ to exact posterior")
    # both versions are standalone figures in the manuscript, so both
    # carry the legend -- it used to be drawn only on the unnormalized one
    ax.legend(frameon=False)
    fig.tight_layout()
    style.save(fig, path)
    plt.close(fig)


def figure_B(M_results, path, key="w1", err="sem", show_reference=False):
    """
    DF-ETLD accuracy vs ensemble size, one curve per alpha.

    Only the derivative-free method is drawn. The ALDI exact-gradient rung and
    the slope -1/2 Monte Carlo floor are still computed by the sweep and stored
    in bias_vs_ensemble_size.json, and can be overlaid with show_reference=True. Keep them to
    hand: they are what distinguishes a genuine bias plateau from the finite-
    sampling floor, since a flat DF-ETLD curve means nothing unless you know
    the floor beneath it is still descending.
    """
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    # alpha is ordinal, so a single-hue ramp (light -> dark), not
    # unrelated categorical hues
    colors = style.levels(len(M_results))
    all_M = sorted({M for byM in M_results.values() for M in byM})

    for i, (a, byM) in enumerate(sorted(M_results.items())):
        Ms = sorted(byM)
        res = [byM[M] for M in Ms]
        c = colors[i]
        _errbar(ax, Ms, res, "dfetld", key=key, err=err, color=c,
                label=rf"$\alpha={a}$")
        if show_reference:
            _errbar(ax, Ms, res, "aldi", key=key, err=err, color=c,
                    ls="--", marker="s", ms=4, alpha=0.6)

    if show_reference and all_M:
        # slope -1/2 guide: pure Monte Carlo error of an M-particle ensemble
        ref_a = sorted(M_results)[0]
        y0 = M_results[ref_a][all_M[0]]["aldi"][key]["mean"]
        xs = np.array([all_M[0], all_M[-1]], dtype=float)
        ax.plot(xs, y0 * (xs / xs[0]) ** -0.5, "k:", lw=1,
                label=r"slope $-1/2$ (MC floor)")
        ax.plot([], [], "k-", marker="o", label=style.label("dfetld"))
        ax.plot([], [], "k--", marker="s", ms=4, alpha=0.6,
                label="ALDI (exact gradient)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Ticks on the ensemble sizes actually run, as plain integers -- not 10^1.
    _plain_log_axis(ax, "x", all_M)
    _plain_log_axis(ax, "y")
    ax.set_xlim(min(all_M) * 0.85, max(all_M) * 1.18)
    ax.set_xlabel(r"ensemble size $M$ (particles)")
    ax.set_ylabel(r"$W_1$ to exact posterior" if key == "w1"
                  else r"$W_1$ / posterior std")
    ax.legend(frameon=False, fontsize=11, title="nonlinearity")
    ax.legend_.get_title().set_fontsize(11)
    fig.tight_layout()
    style.save(fig, path)
    plt.close(fig)
