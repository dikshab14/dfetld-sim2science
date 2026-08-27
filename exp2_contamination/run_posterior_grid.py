"""
exp2_contamination/run_posterior_grid.py
=================================
Robustness under Huber epsilon-contamination, which follows the
misspecified normal location experiment of Matsubara et al. (2022b).

    y_i ~ (1 - eps) N(theta*, 1) + eps N(z, 1),      theta* = 1

  row 1: z = 10 fixed, eps in {0, 0.1, 0.2}
  row 2: eps = 0.1 fixed, z in {1, 10, 20}

with the kernel-score column produced by DF-ETLD (ours, derivative-free)
and the energy-score column by adSGLD (gradient-based baseline).

Every panel pools N_SEEDS independent datasets, as the L96 marginals do,
so the curves carry within-dataset uncertainty AND dataset-to-dataset
variability together.

SETUP PROVENANCE. The arXiv figure does not state N, but the
standard-Bayes column determines it: for a weak prior the conjugate
posterior is ~ N(ybar, 1/N) with E[ybar] = (1-eps) theta* + eps z,
giving peaks at 1.0 / 1.9 / 2.8 (row 1) -- exactly the published peak
locations -- and peak height ~ sqrt(N / 2 pi) ~ 4.0 at N = 100, matching
the published height. So N = 100, sigma = 1, theta* = 1. Verified, not
assumed. The prior is this project's own: theta ~ N(2, 1), as in
run_contamination.py.

LEARNING RATES. The three posteriors are different objects, so their raw
concentrations are not comparable; a method looks spuriously "robust" if
its posterior is merely wider. Each generalized posterior's w is
therefore CALIBRATED ONCE on CLEAN data (eps = 0) so that its posterior
sd matches standard Bayes there, then FROZEN for every panel. Any
difference across panels is then robustness, not width.

    python run_posterior_grid.py            # 10 seeds x 9 panels
    python run_posterior_grid.py --demo     # 3 seeds, smaller ensembles/chains
"""

import argparse
import json
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(
    _os.path.abspath(__file__)), '..'))
from common import style
style.apply()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.dfetld import run_dfetld

# ---- experiment configuration (see SETUP PROVENANCE above) --------------
THETA_STAR = 1.0
SIGMA = 1.0
N_DATA = 100
PRIOR_MEAN, PRIOR_STD = 2.0, 1.0           # this project's prior
EPS_ROW = [0.0, 0.1, 0.2]                  # row 1: z = 10 fixed
Z_ROW1 = 10.0
Z_ROW = [1.0, 10.0, 20.0]                  # row 2: eps = 0.1 fixed
EPS_ROW2 = 0.1
N_SEEDS = 10
XLIM = (-1.0, 5.5)  # wide frame; note the generalized posteriors
                    # occupy only ~1/6 of it, which is the point
SEED0 = 4000


def generate_data(eps, z, rng, n=N_DATA):
    """y ~ (1-eps) N(theta*, 1) + eps N(z, 1); the outlier count is exact
    (round(eps n)) so panels differ by the contamination itself and not by
    a random number of outliers."""
    n_out = int(round(eps * n))
    y = THETA_STAR + SIGMA * rng.standard_normal(n)
    if n_out:
        y[:n_out] = z + SIGMA * rng.standard_normal(n_out)
    return y[:, None]                      # (N, 1): N observations, dim 1


# ---- method 1: standard Bayes (conjugate, exact) ------------------------
def standard_bayes(y, rng, n_samples=4000):
    """Exact conjugate normal posterior, sampled i.i.d. so all three
    columns are rendered through the identical KDE pipeline (these draws
    are exact -- no MCMC error is introduced)."""
    n = len(y)
    prec = 1.0 / PRIOR_STD ** 2 + n / SIGMA ** 2
    mean = (PRIOR_MEAN / PRIOR_STD ** 2 + y.sum() / SIGMA ** 2) / prec
    return mean + np.sqrt(1.0 / prec) * rng.standard_normal(n_samples)


# ---- method 2: DF-ETLD on the kernel-score (MMD) posterior --------------
def model_gamma(rng, n=400):
    """
    Median-heuristic bandwidth on SIMULATOR DRAWS, computed once and
    frozen for the whole experiment.
    """
    x = SIGMA * rng.standard_normal(n)
    d = np.abs(x[:, None] - x[None, :])
    return float(np.median(d[np.triu_indices(n, k=1)]))


def df_etld(y, gamma, w, rng, M=50, J=20, n_steps=2000, dt=5e-3):
    """
    Derivative-free ensemble sampler; the simulator is x = theta + sigma u,
    so no gradient is ever formed. beta = w * N follows the
    sum-over-observations convention of the generalized posterior.

    ENSEMBLE SIZE. M = 50, chosen by measurement: sweeping
    M in {10, 20, 50, 100, 200} on shared datasets (so dataset variability
    cancels and only the sampler effect remains).
    """
    n = len(y)

    def simulate(theta, U):                # theta (M,1), U (J,)
        return theta[:, None, :] + SIGMA * U[None, :, None]

    samples, diag = run_dfetld(
        simulate, y, [PRIOR_MEAN], [PRIOR_STD], M=M, J=J, D=1,
        beta=w * n, gamma=gamma, dt=dt, n_steps=n_steps,
        burn_in=n_steps // 2, thin=5, rng=rng,
        draw_latents=lambda r: r.standard_normal(J))
    return samples.ravel()


# ---- method 3: adSGLD on the energy-score posterior ---------------------
def adsgld_energy(y, w, rng, m=100, n_steps=20000, step=2e-3,
                  burn_frac=0.5, thin=5):
    """
    SGLD on pi(theta) prop. prior * exp(-w sum_i ES(P_theta, y_i)), with
    the EXACT gradient.
    """
    yv = y.ravel()
    burn = int(burn_frac * n_steps)
    theta = float(np.median(yv))
    kept = []
    for k in range(n_steps):
        x = theta + SIGMA * rng.standard_normal(m)
        g_es = (2.0 / m) * np.sign(x[:, None] - yv[None, :]).sum()
        grad = w * g_es + (theta - PRIOR_MEAN) / PRIOR_STD ** 2
        theta = (theta - 0.5 * step * grad
                 + np.sqrt(step) * rng.standard_normal())
        if k >= burn and (k - burn) % thin == 0:
            kept.append(theta)
    return np.array(kept)


# ---- learning-rate calibration on CLEAN data ---------------------------
def calibrate(fn, target_sd, w0, seed, n_iter=3):
    """Scale w so the eps=0 posterior sd matches standard Bayes. For a
    locally quadratic log-posterior sd ~ 1/sqrt(w), so w <- w (sd/target)^2
    converges in a couple of passes."""
    w = w0
    for _ in range(n_iter):
        s = float(np.std(fn(w, np.random.default_rng(seed))))
        w = w * (s / target_sd) ** 2
        if not (1e-4 < w < 1e5):
            raise RuntimeError(
                f"learning-rate calibration diverged (w={w:.3g}, measured "
                f"sd={s:.3g} vs target {target_sd:.3g}) -- the sampler is "
                "not equilibrating; do not trust the panels")
    return w


PANELS = ([("eps", e, Z_ROW1) for e in EPS_ROW]
          + [("z", EPS_ROW2, z) for z in Z_ROW])


def run_panel_seed(task, w_df, w_ad, cfg):
    """One (panel, seed): all three posteriors on the same dataset."""
    i, seed = task
    kind, eps, z = PANELS[i]
    rng = np.random.default_rng(SEED0 + 1000 * seed + i)
    y = generate_data(eps, z, rng)
    g = model_gamma(rng)
    return (i, g,
            standard_bayes(y, rng),
            df_etld(y, g, w_df, rng, M=cfg["M"], J=cfg["J"],
                    n_steps=cfg["n_steps"]),
            adsgld_energy(y, w_ad, rng, n_steps=cfg["ad_steps"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cfg = ({"M": 25, "J": 10, "n_steps": 800, "ad_steps": 5000}
           if args.demo else
           {"M": 50, "J": 20, "n_steps": 2000, "ad_steps": 20000})
    n_seeds = 3 if args.demo else N_SEEDS

    y_clean = generate_data(0.0, Z_ROW1, np.random.default_rng(SEED0))
    gamma_c = model_gamma(np.random.default_rng(SEED0))
    sd_target = float(np.std(standard_bayes(y_clean,
                                            np.random.default_rng(SEED0))))
    print(f"gamma (clean) = {gamma_c:.3f}   target sd = {sd_target:.4f}",
          flush=True)
    w_df = calibrate(lambda w, r: df_etld(y_clean, gamma_c, w, r,
                                          M=cfg["M"], J=cfg["J"],
                                          n_steps=cfg["n_steps"]),
                     sd_target, 1.0, SEED0 + 1)
    w_ad = calibrate(lambda w, r: adsgld_energy(y_clean, w, r,
                                                n_steps=cfg["ad_steps"]),
                     sd_target, 1.0, SEED0 + 2)
    print(f"calibrated w: kernel (DF-ETLD) {w_df:.3f}   "
          f"energy (adSGLD) {w_ad:.3f}", flush=True)

    tasks = [(i, s) for i in range(len(PANELS)) for s in range(n_seeds)]
    with Pool(5) as pool:
        res = pool.map(partial(run_panel_seed, w_df=w_df, w_ad=w_ad,
                               cfg=cfg), tasks)

    acc = {i: {"sb": [], "df": [], "ad": [], "gamma": []}
           for i in range(len(PANELS))}
    for i, g, sb, df, ad in res:
        acc[i]["sb"].append(sb)
        acc[i]["df"].append(df)
        acc[i]["ad"].append(ad)
        acc[i]["gamma"].append(g)

    out = {}
    for i, (kind, eps, z) in enumerate(PANELS):
        key = f"{kind}_eps{eps}_z{z}"
        out[key] = {m: np.concatenate(acc[i][m]) for m in ("sb", "df", "ad")}
        out[key]["gamma"] = float(np.mean(acc[i]["gamma"]))
        print(f"  {key:18} gamma {out[key]['gamma']:5.2f}   mean  "
              f"SB {out[key]['sb'].mean():6.3f}   "
              f"kernel {out[key]['df'].mean():6.3f}   "
              f"energy {out[key]['ad'].mean():6.3f}", flush=True)

    np.savez_compressed(
        os.path.join(args.outdir, "contamination_posterior_grid_samples.npz"),
        **{f"{k}__{m}": np.atleast_1d(v) for k, d in out.items()
           for m, v in d.items()})
    json.dump({"gamma_clean": gamma_c, "w_kernel": w_df, "w_energy": w_ad,
               "sd_target": sd_target, "theta_star": THETA_STAR,
               "N": N_DATA, "n_seeds": n_seeds,
               "prior": [PRIOR_MEAN, PRIOR_STD],
               "means": {k: {m: float(np.mean(d[m]))
                             for m in ("sb", "df", "ad")}
                         for k, d in out.items()}},
              open(os.path.join(args.outdir, "contamination_posterior_grid_config.json"), "w"),
              indent=1)
    make_figure(out, os.path.join(args.outdir, "contamination_posterior_grid.png"),
                n_seeds)
    print(f"\n-> {args.outdir}/contamination_posterior_grid.png")


def _kde(x, grid):
    """Gaussian KDE, Silverman bandwidth -- the same estimator the exp2
    marginals use, and the rendering the reference figure describes."""
    x = np.asarray(x)
    bw = max(0.9 * x.std() * len(x) ** (-0.2), 1e-3)
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(1) / (len(x) * bw * np.sqrt(2 * np.pi))


def make_figure(out, path, n_seeds):
    """Styling matches the exp2 marginals: line-only KDEs at lw 1.6,
    tab10 colours, dashed black theta*, dotted grey prior, frameless
    small legends, no y ticks."""
    # the "ad" column is PLAIN SGLD on the energy score (no adaptive
    # preconditioner; in D = 1 it would only rescale the step), so it is
    # labelled SGLD-ES. Colour is unchanged.
    cols = [("sb", style.label("bayes")),
            ("df", style.label("dfetld")),
            ("ad", style.label("sgld_es"))]
    # eps and z are ordinal, so one hue light -> dark. The method hues
    # were used here before, which made teal mean "standard Bayes" in
    # every other figure and "eps = 0" in this one.
    colors = style.levels(3)
    grid = np.linspace(*XLIM, 600)
    prior = (np.exp(-0.5 * ((grid - PRIOR_MEAN) / PRIOR_STD) ** 2)
             / (PRIOR_STD * np.sqrt(2 * np.pi)))

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.0), sharex=True,
                             sharey=True)
    ymax = 0.0
    for j, (mk, title) in enumerate(cols):
        for e, c in zip(EPS_ROW, colors):
            k = f"eps_eps{e}_z{Z_ROW1}"
            f = _kde(out[k][mk], grid)
            ymax = max(ymax, f.max())
            axes[0][j].plot(grid, f, color=c, lw=1.6,
                            label=rf"$\epsilon={e:g}$" if j == 2 else None)
        for z, c in zip(Z_ROW, colors):
            k = f"z_eps{EPS_ROW2}_z{z}"
            f = _kde(out[k][mk], grid)
            ymax = max(ymax, f.max())
            axes[1][j].plot(grid, f, color=c, lw=1.6,
                            label=rf"$z={z:g}$" if j == 2 else None)
        axes[0][j].set_title(title)
        for r in (0, 1):
            ax = axes[r][j]
            ax.plot(grid, prior, color="0.6", lw=0.8, ls=":",
                    label="prior" if (j == 0 and r == 0) else None)
            ax.axvline(THETA_STAR, color="k", lw=1.3, ls="--",
                       label=r"true $\theta^*$" if (j == 0 and r == 0)
                       else None)
            ax.set_xlim(*XLIM)
            ax.set_yticks([])
        axes[1][j].set_xlabel(r"$\theta$")
    # shared y with headroom so no curve touches the frame
    axes[0][0].set_ylim(0.0, 1.15 * ymax)

    axes[0][0].set_ylabel("generalized posterior\n" rf"$z={Z_ROW1:g}$")
    axes[1][0].set_ylabel("generalized posterior\n"
                          rf"$\epsilon={EPS_ROW2:g}$")
    # sweep legends go in the right-hand column, where the robust
    # posteriors leave the panel empty (as in the reference figure)
    axes[0][2].legend(frameon=False, fontsize=11, loc="upper right")
    axes[1][2].legend(frameon=False, fontsize=11, loc="upper right")
    # prior / theta* key: shared by every panel, so it belongs to the
    # FIGURE, centred beneath it, rather than inside one panel
    ref_handles, ref_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(ref_handles, ref_labels, frameon=False, fontsize=12,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.055, 1, 1), w_pad=2.0)   # room for the figure legend
    style.save(fig, path)
    plt.close(fig)


if __name__ == "__main__":
    main()
