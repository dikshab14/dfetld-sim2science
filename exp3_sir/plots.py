"""
exp3_sir/plots.py
======================
Figures mirroring Kelly et al. (2024) Figures 7 and 8 for the misspecified
SIR benchmark.

sir_posteriors  -- bivariate posterior contours for (eta, beta) with
            univariate marginals on the sides, one panel per method
            (DF-ETLD, SMC-ABC, adSGLD), truth marked with a cross.
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
from scipy.stats import gaussian_kde

import sir_model as S

ORDER = ["dfetld", "abc", "adsgld"]
# this experiment's adSGLD runs on the SAME kernel-score (MMD) target as
# DF-ETLD, so it carries the -KS suffix exp2 and exp3 use; a bare
# "adSGLD" would read as a fourth variant across the manuscript


def _key(k):
    """Map this experiment's local key onto the house palette key."""
    return "adsgld_ks" if k == "adsgld" else k


class _ByMethod(dict):
    """Label/colour lookup that works for ANY method key, not just the
    three in ORDER -- so a variant run (the energy-score arm, say) can be
    plotted through the same routines without redefining the palette."""

    def __init__(self, fn):
        self._fn = fn

    def __missing__(self, k):
        return self._fn(_key(k))


LABELS = _ByMethod(style.label)
COLORS = _ByMethod(style.color)
# SIR-LOCAL LABEL OVERRIDE. This experiment's gradient-based arm is
# PLAIN SGLD: it uses finite-difference gradients with no adaptive
# preconditioner (a preconditioned Langevin diffusion is invariant
# only with the div(G) correction, which RMSprop omits; here that
# bias was ~3x the posterior sd). Calling it adSGLD would claim a
# component it does not have, so the SIR figures say SGLD. The
# Lorenz '96 arm IS preconditioned and keeps its adSGLD label --
# hence the override is local to this module, not in common/style.
LABELS["adsgld_es"] = style.label("sgld_es")


def _kde_grid(x, y, w, lim, n=220):
    gx = np.linspace(lim[0][0], lim[0][1], n)
    gy = np.linspace(lim[1][0], lim[1][1], n)
    GX, GY = np.meshgrid(gx, gy)
    try:
        k = gaussian_kde(np.vstack([x, y]), weights=w)
        Z = k(np.vstack([GX.ravel(), GY.ravel()])).reshape(n, n)
    except Exception:
        Z = np.zeros((n, n))
    return gx, gy, Z


def sir_posteriors(post, path, lim=None, credible=None,
                   order=None, truth_on_marginals=False):
    """
    post: dict method -> (theta (n,2) as (beta, eta), weights or None)
    Layout per panel: central contour (eta on x, beta on y) with marginal
    densities above and to the right, truth marked.
    """
    order = ORDER if order is None else order
    if lim is None:
        allb = np.concatenate([p[0][:, 0] for p in post.values()])
        alle = np.concatenate([p[0][:, 1] for p in post.values()])
        qe = np.percentile(alle, [0.2, 99.8])
        qb = np.percentile(allb, [0.2, 99.8])
        pad = 0.04
        lim = ((max(0.0, qe[0] - pad), min(0.5, qe[1] + pad)),
               (max(0.0, qb[0] - pad), min(0.5, qb[1] + pad)))

    fig = plt.figure(figsize=(11.0, 3.9))
    for i, key in enumerate(order):
        th, w = post[key]
        beta, eta = th[:, 0], th[:, 1]
        w = np.ones(len(th)) / len(th) if w is None else w
        base = 0.06 + i * 0.325
        axc = fig.add_axes([base, 0.13, 0.205, 0.56])
        axt = fig.add_axes([base, 0.72, 0.205, 0.17], sharex=axc)
        axr = fig.add_axes([base + 0.21, 0.13, 0.055, 0.56], sharey=axc)

        gx, gy, Z = _kde_grid(eta, beta, w, lim)
        # the prior support is the triangle beta > eta; mask outside it so
        # the KDE does not draw contours across the hard boundary
        GX, GY = np.meshgrid(gx, gy)
        # ZERO outside the prior triangle, not NaN. With NaN the contour
        # simply stops at the mask edge and the curve appears broken;
        # with 0 the level set closes along the boundary, which is what a
        # density truncated at beta = eta actually looks like.
        Z = np.where(GY > GX, Z, 0.0)
        # Contour LEVELS as fractions of the peak density, not the
        # automatic 6 levels spanning [min, max] of the masked array.
        zmax = float(np.nanmax(Z)) if np.isfinite(Z).any() else 0.0
        if credible is not None:
            # CREDIBLE contours: the level enclosing a given posterior
            # mass, found by sorting the density on the grid and walking
            # down until that fraction of the total is accumulated. These
            # answer "is the truth covered at 95%?" directly, which
            # peak-density fractions do not.
            flat = np.sort(Z.ravel())[::-1]
            csum = np.cumsum(flat) / flat.sum()
            lv = sorted(float(flat[np.searchsorted(csum, c)])
                        for c in credible)
            lv = [v for i, v in enumerate(lv) if i == 0 or v > lv[i - 1]]
            if lv:
                # no inline labels: they break the contour they sit on,
                # which is the thing the reader is trying to follow
                axc.contour(gx, gy, Z, levels=lv, colors=COLORS[key],
                            linewidths=1.8)
        elif zmax > 0:
            axc.contour(gx, gy, Z, levels=np.array([.1, .3, .5, .7, .9])
                        * zmax, colors=COLORS[key], linewidths=1.8)
        axc.plot(S.ETA_TRUE, S.BETA_TRUE, "kx", ms=5, mew=1.4,
                 label=r"true $(\eta,\beta)$" if i == 0 else None)
        if i == 0:
            # upper left is the one corner the diagonal ridge never
            # reaches in any of the three panels
            axc.legend(frameon=False, loc="upper left", fontsize=11)
        axc.set_xlim(*lim[0]); axc.set_ylim(*lim[1])
        axc.set_xlabel(r"$\eta$")
        if i == 0:
            axc.set_ylabel(r"$\beta$")

        for ax, v, lo_hi, vert in ((axt, eta, lim[0], False),
                                   (axr, beta, lim[1], True)):
            g = np.linspace(lo_hi[0], lo_hi[1], 200)
            try:
                dens = gaussian_kde(v, weights=w)(g)
            except Exception:
                dens = np.zeros_like(g)
            # truth_on_marginals: the cross on the bivariate panel marks
            # the PAIR, which answers "is (eta, beta) covered" but not
            # "is eta covered" --
            # those are different questions. 
            if vert:
                axr.plot(dens, g, color=COLORS[key], lw=2.0)
                if truth_on_marginals:
                    axr.axhline(S.BETA_TRUE, color=style.color("truth"),
                                lw=1.4, ls="--", zorder=1)
                axr.set_xticks([])
                axr.tick_params(labelleft=False)
            else:
                axt.plot(g, dens, color=COLORS[key], lw=2.0)
                if truth_on_marginals:
                    axt.axvline(S.ETA_TRUE, color=style.color("truth"),
                                lw=1.4, ls="--", zorder=1)
                axt.set_yticks([])
                axt.tick_params(labelbottom=False)
        axt.set_title(LABELS[key])
    style.save(fig, path)
    plt.close(fig)


def sir_observed(y_obs, path, zoom_days=42):
    """
    One simulated series from the TRUE (misspecified) DGP -- the analogue
    of Kelly et al. Fig. 16.

    Left: the full 365-day series, as they print it. Right: a short window
    around the peak, because the weekly sawtooth is what makes this DGP
    misspecified (Mondays +10%, weekends -5%) and it is not legible at a
    365-day horizon. Two separate panels rather than an inset -- an inset
    needs connector lines across the curve and obscures the data it sits
    on top of.
    """
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(9.4, 3.4),
                                  gridspec_kw={"width_ratios": [1.7, 1]})
    days = np.arange(len(y_obs))
    pk = int(np.argmax(y_obs))

    ax.plot(days, y_obs, color=style.color("observed"), lw=1.8)
    ax.set_xticks([0, pk, len(y_obs)])
    ax.set_yticks([0, y_obs.max() / 2, y_obs.max()])
    ax.set_yticklabels(["0", f"{y_obs.max() / 2:.0f}",
                        f"{y_obs.max():.0f}"])
    ax.set_xlabel("Days")
    ax.set_ylabel("Number of infected")
    ax.set_xlim(0, len(y_obs))
    ax.set_ylim(0, 1.03 * y_obs.max())

    a0 = max(0, pk - zoom_days // 2)
    b0 = min(len(y_obs), a0 + zoom_days)
    axz.plot(days[a0:b0], y_obs[a0:b0],
             color=style.color("observed"), lw=2.4)
    mon = [d for d in range(a0, b0) if d % 7 == 0]
    # the annotation colour, not a method colour: coral is adSGLD-KS
    # everywhere else in the paper
    axz.plot(mon, y_obs[mon], "o", ms=7, color=style.ANNOT, lw=0,
             label="Mondays (+10%)")
    axz.set_xlabel("Days")
    axz.set_xlim(a0, b0 - 1)
    axz.legend(frameon=False, fontsize=12)
    fig.tight_layout()
    style.save(fig, path)
    plt.close(fig)


def cost_table(rows, path_md, path_json):
    """
    rows: list of dicts with keys method, sim_calls, wallclock_s, notes.
    """
    import json
    json.dump(rows, open(path_json, "w"), indent=1)
    with open(path_md, "w") as f:
        f.write("| method | simulator calls | wall-clock (s) | notes |\n")
        f.write("|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['method']} | {r['sim_calls']:,} | "
                    f"{r['wallclock_s']:.0f} | {r['notes']} |\n")
    return open(path_md).read()
