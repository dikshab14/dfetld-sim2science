"""
common/style.py
===============
One place for figure styling, so every experiment's plots are consistent
and manuscript-legible.

COLOUR IS ASSIGNED BY METHOD, NOT BY POSITION. A method keeps its hue in
every figure it appears in, so a reader who learns "blue = DF-ETLD" in
Figure 1 does not relearn it in Figure 5. Colours are never cycled by
plot order.

One hue FAMILY per method, so nothing is distinguished by lightness
alone -- except SMC-ABC and its 6-summary variant, which are the same
algorithm with a different distance and therefore share the magenta
family at two steps.

The palette was validated rather than eyeballed (lightness band, chroma
floor, colour-vision-deficiency separation on adjacent pairs,
normal-vision floor, contrast against the surface). Each set of methods
that actually co-occurs in a figure was checked as its own palette:

    appendix study    DF-ETLD, ALDI                     all checks pass
    exp1_l96          DF-ETLD, adSGLD-KS/ES, ABC, ABCs  all pass, worst CVD dE 12.0
    exp2_contam.      DF-ETLD, adSGLD-KS/ES, std Bayes  all pass, worst CVD dE 16.0
    exp3_sir          DF-ETLD, adSGLD, ABC              all pass, worst CVD dE 12.8

DF-ETLD's blue was darkened until it cleared 3:1 contrast against white:
it is the method the reader should be able to pick out first, and thin
lines in print need the contrast.

Titles are deliberately absent from every figure -- the manuscript
caption carries the description.
"""

import matplotlib as mpl

# ---- method -> colour, fixed across every experiment -------------------
DF_ETLD = "#1f5fa8"      # blue    -- ours (derivative-free)
ADSGLD_KS = "#e8613c"    # coral   -- gradient-based, kernel score
ADSGLD_ES = "#7b3fa0"    # purple  -- gradient-based, energy score
ABC = "#c2185b"          # magenta -- SMC-ABC
ABC_SUMM = "#ef92b4"     # light magenta -- same method, summary distance
BAYES = "#00949e"        # teal    -- exact / likelihood-based reference
TRUTH = "#111111"        # near-black -- observed data and true parameter
ANNOT = "#b8860b"        # dark gold -- RESERVED for annotating the data
                         # (e.g. the weekday reporting artefact). Held out
                         # of the method palette on purpose: an annotation
                         # drawn in coral reads as "this is adSGLD-KS".

METHOD_COLOR = {
    # ours (derivative-free)
    "dfetld": DF_ETLD, "dfr": DF_ETLD,
    # gradient-based baseline, by scoring rule
    "adsgld": ADSGLD_KS, "adk": ADSGLD_KS, "adsgld_ks": ADSGLD_KS,
    "aldi": ADSGLD_KS,          # the appendix study's oracle rung:
                                # also gradient-based
    "ad": ADSGLD_ES, "adsgld_es": ADSGLD_ES,
    # PLAIN-SGLD arms (exp2_contamination, exp3_sir): same hues, but
    # named without the "ad" prefix -- see METHOD_LABEL below.
    "sgld_ks": ADSGLD_KS, "sgld_es": ADSGLD_ES,
    # likelihood-free competitor
    "abc": ABC, "abcs": ABC_SUMM,
    # exact references
    "bayes": BAYES, "bayes_p": BAYES,
    "truth": TRUTH, "observed": TRUTH,
}

METHOD_LABEL = {
    "dfetld": "DF-ETLD", "dfr": "DF-ETLD",
    "aldi": "ALDI (exact gradient)",
    "adsgld": "adSGLD", "adk": "adSGLD-KS", "adsgld_ks": "adSGLD-KS",
    "ad": "adSGLD-ES", "adsgld_es": "adSGLD-ES",
    # exp2_contamination and exp3_sir run PLAIN SGLD -- no adaptive
    # preconditioner -- so their figures use these names. (exp1_l96 IS
    # RMSprop-preconditioned and keeps the adSGLD labels above.)
    "sgld_ks": "SGLD-KS", "sgld_es": "SGLD-ES",
    "abc": "SMC-ABC", "abcs": "SMC-ABC (summaries)",
    "bayes": "standard Bayes", "truth": "truth", "observed": "observed",
}


# ---- ordinal levels (alpha, epsilon, z, ...) ---------------------------
# A level is a MAGNITUDE, so it gets one hue light -> dark, never the
# categorical method hues: teal reads as "standard Bayes" everywhere else,
# and it must not also mean "eps = 0" in one figure.
LEVEL_RAMP = ["#bcd3ec", "#87b0da", "#4f8ac6", "#1f5fa8", "#123a68"]


def levels(n, dark=True):
    """`n` colours spread over the whole ramp, light -> dark.

    Spread, not the first n steps: with n = 3 the first-n rule uses only
    the pale end, and #bcd3ec on white is too faint to read as a line.
    dark=False keeps the palest step for filled areas, which need it.
    """
    lo = 1 if (dark and n <= 3) else 0
    if n == 1:
        return [LEVEL_RAMP[-1]]
    span = len(LEVEL_RAMP) - 1 - lo
    return [LEVEL_RAMP[lo + round(i * span / (n - 1))] for i in range(n)]


def color(key):
    return METHOD_COLOR.get(key, "#52514e")


def label(key):
    return METHOD_LABEL.get(key, key)


def apply(scale=1.0):
    """Manuscript-legible defaults. Call once at import time in a plots
    module. `scale` multiplies every font size, for figures that will be
    reproduced small."""
    mpl.rcParams.update({
        "font.size": 13 * scale,
        "axes.labelsize": 16 * scale,
        "axes.titlesize": 16 * scale,
        "xtick.labelsize": 14 * scale,
        "ytick.labelsize": 14 * scale,
        "legend.fontsize": 14 * scale,
        "figure.titlesize": 16 * scale,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.0,
        "axes.linewidth": 1.0,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


def save(fig, path, **kw):
    """savefig that first GROWS the canvas if an axis label does not fit.

    Matplotlib clips text at the figure canvas, and bbox_inches="tight"
    then measures the CLIPPED extent -- so a rotated y label taller than
    its axes is silently cut off and no warning is raised. Measured: a
    label needing 3.87 in on a 2.0 in canvas saved at 1.82 in, losing the
    end of the text. get_window_extent DOES report the true size, so the
    fix is to compare label extent against axes extent and enlarge the
    figure by the deficit before saving.
    """
    for _ in range(3):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        dpi = float(fig.dpi)
        grow_h = grow_w = 0.0
        for ax in fig.axes:
            box = ax.get_window_extent(r)
            if ax.yaxis.label.get_text():
                e = ax.yaxis.label.get_window_extent(r)
                grow_h = max(grow_h, (e.height - box.height) / dpi)
            if ax.xaxis.label.get_text():
                e = ax.xaxis.label.get_window_extent(r)
                grow_w = max(grow_w, (e.width - box.width) / dpi)
        if grow_h <= 0.02 and grow_w <= 0.02:
            break
        w, h = fig.get_size_inches()
        # grow the canvas ONLY -- axes positions are stored as figure
        # fractions, so they scale with it. Re-running tight_layout here
        # would discard any rect= the caller used to reserve space for a
        # figure legend, and drop the legend on top of the axis labels.
        fig.set_size_inches(w + max(0.0, grow_w) + 0.15,
                            h + max(0.0, grow_h) + 0.15)
    fig.savefig(path, **kw)
