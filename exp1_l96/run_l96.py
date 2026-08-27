"""
exp1_l96/run_l96.py
===============================
Driver for the WELL-SPECIFIED stochastic Lorenz96 experiment: the data
are generated from the same model the samplers fit, at
theta* = (theta1, theta2, sigma_e) = (2, 0.8, 1.7) with phi fixed at 0
(3 parameters, l96.generate_observation_ws), from the fixed known
initial state l96.X0_FIXED, n = 10 series x 5 observation sets. No model error is present, so a gap
between methods here is the SAMPLER, not robustness.

Three samplers:

    DF-ETLD    ours, natural theta space, uniform box by reflection
               (no reparameterization: a squashing map would compose the
               simulator with extra curvature, feeding the appendix
               study's linearization-bias mechanism)
    SMC-ABC    Euclidean distance, adaptive tolerance, benchmark config
               (1000 particles x m=10 x 25 generations = 250k calls)
    adSGLD-ES  energy-score posterior via preconditioned SGLD on exact
               RK4 forward sensitivities, benchmark config
               (w = 3.4882, m = 200, 20000 steps = 4e6 calls)
    adSGLD-KS  the CONTROL: same sampler on DF-ETLD's kernel-score
               target (same w, same gamma), so the pair isolates the
               sampler from the scoring rule

Budgets are deliberately NOT equalized here: these are the benchmark
configurations, which are themselves not budget-matched (ABC 250k vs
SG-MCMC 4e6 calls). Per-method call counts are recorded in
results_*.json.

    python run_l96.py --demo     # smoke run, 2 sets (M=30, J=8)
    python run_l96.py            # 5 sets (M=100, J=10, 500k calls/method)

Each observation set is fully determined by its own rng (12000 + s), so
parallelizing over sets is bit-identical to this sequential loop.
"""

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(
    _os.path.abspath(__file__)), '..'))
from common import style
style.apply()

import l96
import methods
from methods import median_heuristic_from_sims

# labels and colours come from common/style so a method keeps its hue
# across every experiment
METHOD_STYLE = {k: (style.label(k), style.color(k))
                for k in ("dfr", "abc", "abcs", "ad", "adk")}
# METHODS is the PLOTTED set; ALL_METHODS is what is computed and stored.
# Both SMC-ABC variants are run and saved (raw-data distance and the
# 6-summary distance, the two ABC rows of Table 1), but only the
# full-data one is drawn, to keep the figure legible.
# "adk" is the CONTROL, not another baseline: adSGLD run on the kernel
# score at DF-ETLD's own w and gamma targets the IDENTICAL generalized
# posterior (sum_i S_k = n MMD^2 + const), so dfr-vs-adk isolates the
# SAMPLER while adk-vs-ad isolates the SCORING RULE. Comparing dfr with
# ad alone confounds the two, which is exactly the comparison the main
# claim ("derivative-free costs little") must not rest on.
METHODS = ["dfr", "adk", "abc", "ad"]
ALL_METHODS = ["dfr", "adk", "abc", "abcs", "ad"]
BOX_METHODS = {"dfr", "adk", "abc", "abcs", "ad"}
# All methods share the uniform box prior.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def _kde1d(x, grid, lo, hi, weights=None, max_n=20000, reflect=True):
    """Boundary-reflected Gaussian KDE on [lo, hi]; Silverman bandwidth on
    the weighted effective sample size. Reflection keeps mass piled at a
    box face rendered AT the face instead of leaking outside."""
    x = np.asarray(x, float)
    if weights is not None:
        w = np.asarray(weights, float)
        w = w / w.sum()
    else:
        w = np.full(len(x), 1.0 / len(x))
    if len(x) > max_n:
        idx = np.random.default_rng(0).choice(len(x), max_n, p=w)
        x = x[idx]
        w = np.full(max_n, 1.0 / max_n)
    n_eff = 1.0 / np.sum(w ** 2)
    sd = np.sqrt(np.sum(w * (x - np.sum(w * x)) ** 2))
    bw = max(0.9 * sd * n_eff ** (-0.2), (hi - lo) / 400.0)

    def dens(g):
        z = (g[:, None] - x[None, :]) / bw
        return (np.exp(-0.5 * z ** 2) @ w) / (bw * np.sqrt(2.0 * np.pi))

    if not reflect:          # unbounded support
        return dens(grid)
    return dens(grid) + dens(2.0 * lo - grid) + dens(2.0 * hi - grid)


def interval_coverage(samples, weights, tstar, levels=(0.5, 0.9)):
    """
    Does the credible interval actually contain theta*? Equal-tailed
    intervals from the (optionally weighted) marginal posterior, one flag
    per parameter per level, plus the interval width.

    This is the companion to the point-estimate error: a method can match
    theta* with its posterior MEAN while being badly over- or
    under-confident, and only coverage sees that. With 5 observation sets
    the coverage estimate is coarse (resolution 1/5), so read it as a
    direction, not a calibrated number.
    """
    out = {}
    for lv in levels:
        lo_q, hi_q = 0.5 - lv / 2.0, 0.5 + lv / 2.0
        lo, hi = _wquantile(samples, weights, [lo_q, hi_q])
        out[f"cov{int(lv * 100)}"] = ((tstar >= lo) & (tstar <= hi)).tolist()
        out[f"width{int(lv * 100)}"] = (hi - lo).tolist()
    return out


def _wquantile(x, w, qs):
    """Weighted quantiles, per column. w=None -> unweighted."""
    x = np.asarray(x)
    n, d = x.shape
    w = np.full(n, 1.0 / n) if w is None else np.asarray(w) / np.sum(w)
    out = np.empty((len(qs), d))
    for j in range(d):
        o = np.argsort(x[:, j])
        cw = np.cumsum(w[o])
        cw /= cw[-1]
        out[:, j] = np.interp(qs, cw, x[o, j])
    return out


def run_one_set(s, demo=False):
    """One observation set: data -> three posteriors at matched budget ->
    predictive energy scores + recovery errors."""
    if demo:
        M, Jsim, n_steps, S_pred = 30, 8, 250, 30
        n_pop, abc_gen, ad_m, ad_steps = 200, 8, 50, 300
    else:
        M, Jsim, n_steps, S_pred = 100, 10, 500, 100
        # benchmark configs: ABC 1000 x 10 x 25 = 250k;
        # adSGLD m = 200, 20000 steps = 4e6
        n_pop, abc_gen, ad_m, ad_steps = 1000, 25, 200, 20000

    rng = np.random.default_rng(12000 + s)
    x0, y = l96.generate_observation_ws(rng)          # y: (10, 360)


    steps, K = l96.n_latents()
    th_pp = l96.sample_box(rng, 40)
    X_pp = l96.simulate(th_pp, x0,
                         rng.standard_normal((2, steps, K)))[:, 0]
    gamma = median_heuristic_from_sims(X_pp)

    # w = 30 -> beta = w * n = 300 on the shared 10 series (the
    # production learning rate)
    _t = time.perf_counter()
    th_dfr, calls_df, diag_r = methods.df_etld_reflect(
        y, x0, gamma, rng, M=M, Jsim=Jsim, n_steps=n_steps,
        burn_in=n_steps // 2, w=30.0)
    t_dfr = time.perf_counter() - _t
    # SMC-ABC and adSGLD both at the benchmark configs (see methods.py);
    # note these are NOT equal budgets (ABC 250k vs SG-MCMC 4e6 calls).
    # Call counts for every method are recorded per set so the asymmetry
    # stays visible.
    _t = time.perf_counter()
    th_abc, w_abc, calls_abc = methods.smc_abc(
        y, x0, rng, n_pop=n_pop, m=10, n_gen=abc_gen)
    t_abc = time.perf_counter() - _t
    # second ABC baseline: same algorithm and budget, with the distance
    # on the 6 summary statistics
    th_abcs, w_abcs, _ = methods.smc_abc(
        y, x0, rng, n_pop=n_pop, m=10, n_gen=abc_gen, summaries=True)
    _t = time.perf_counter()
    th_ad, calls_ad, diag_ad = methods.adsgld(
        y, x0, rng, m=ad_m, n_steps=ad_steps, step=5e-3)
    t_ad = time.perf_counter() - _t
    # the control: same sampler as th_ad, pointed at DF-ETLD's exact
    # target -- same kernel score, same gamma, same w = 30 (beta = w n)
    _t = time.perf_counter()
    th_adk, calls_adk, diag_adk = methods.adsgld(
        y, x0, rng, m=ad_m, n_steps=ad_steps, step=5e-3,
        loss="kernel", gamma=gamma, w=30.0)
    t_adk = time.perf_counter() - _t

    es = {}
    draws = {"dfr": th_dfr[rng.choice(len(th_dfr), S_pred)],
             "abc": th_abc[rng.choice(len(th_abc), S_pred, p=w_abc)],
             "abcs": th_abcs[rng.choice(len(th_abcs), S_pred, p=w_abcs)],
             "ad": th_ad[rng.choice(len(th_ad), S_pred)],
             "adk": th_adk[rng.choice(len(th_adk), S_pred)]}
    for name, dr in draws.items():
        pred = methods.posterior_predictive(dr, x0, rng)
        es[name] = methods.energy_score_per_timestep(pred, y)

    tstar = np.array([l96.THETA_STAR[p] for p in l96.PARAM_NAMES])
    means = {"dfr": th_dfr.mean(0),
             "abc": (th_abc * w_abc[:, None]).sum(0),
             "abcs": (th_abcs * w_abcs[:, None]).sum(0),
             "ad": th_ad.mean(0),
             "adk": th_adk.mean(0)}
    row = {"calls_df": calls_df, "calls_abc": calls_abc,
           "calls_ad": calls_ad, "calls_adk": calls_adk,
           "tame_dfr": diag_r["tame_stationary"],
           "tame_ad": diag_ad["tame_stationary"],
           "tame_adk": diag_adk["tame_stationary"],
           # per-method wall-clock, so the cost table does not have to
           # guess: measured inside the run, same hardware, same process
           "time_dfr": t_dfr, "time_abc": t_abc, "time_ad": t_ad,
           "time_adk": t_adk,
           "adsgld_chain_means": diag_ad["chain_means"].tolist(),
           "adsgldk_chain_means": diag_adk["chain_means"].tolist()}
    draws_all = {"dfr": (th_dfr, None), "abc": (th_abc, w_abc),
                 "abcs": (th_abcs, w_abcs), "ad": (th_ad, None),
                 "adk": (th_adk, None)}
    for k, mu in means.items():
        row[f"post_mean_{k}"] = mu.tolist()
        row[f"err_{k}"] = np.abs(mu - tstar).tolist()
        th_k, w_k = draws_all[k]
        for mk, mv in interval_coverage(th_k, w_k, tstar).items():
            row[f"{mk}_{k}"] = mv
    row["post_std_dfr"] = th_dfr.std(0).tolist()
    row["post_std_ad"] = th_ad.std(0).tolist()
    row["post_std_adk"] = th_adk.std(0).tolist()
    # headline number for the main claim, in units a reader can judge:
    # |mean_DF - mean_adSGLD-kernel| / posterior sd, on a shared target
    row["sampler_gap_sd"] = (np.abs(th_dfr.mean(0) - th_adk.mean(0))
                             / th_adk.std(0)).tolist()
    return {"s": s, "row": row, "es": es, "th_dfr": th_dfr,
            "th_ad": th_ad, "th_adk": th_adk,
            "th_abc": th_abc, "w_abc": w_abc,
            "th_abcs": th_abcs, "w_abcs": w_abcs}


def make_plots(results, tag, outdir):
    n_sets = len(results)
    names = METHODS
    tcrit = T95.get(n_sets - 1, 1.96)

    # ---- per-timestep energy score (skip the shared IC column) ----------
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ts = np.arange(1, l96.N_REC)
    for k in names:
        E = np.stack([r["es"][k] for r in results])[:, 1:]
        mu = E.mean(0)
        half = tcrit * E.std(0, ddof=1) / np.sqrt(n_sets)
        lab, c = METHOD_STYLE[k]
        ax.plot(ts, mu, color=c, label=lab)
        ax.fill_between(ts, mu - half, mu + half, color=c, alpha=0.18)
    ax.set_xlabel("timestep")
    # short label: at 16pt the long form is taller than the axes
    # and overflows the canvas. "posterior predictive" belongs
    # in the caption, not on the axis.
    ax.set_ylabel("predictive energy score")
    ax.legend(frameon=False, fontsize=13)
    fig.tight_layout()
    style.save(fig, os.path.join(outdir, f"es_per_timestep_{tag}.png"))
    plt.close(fig)

    # ---- pooled posterior marginals (KDE) -------------------------------
    lo, hi = l96.BOX_LO, l96.BOX_HI
    pooled = {}
    for k in names:
        if k in ("abc", "abcs"):
            th = np.concatenate([r[f"th_{k}"] for r in results])
            wt = np.concatenate([r[f"w_{k}"] / n_sets for r in results])
        else:
            th = np.concatenate([r[f"th_{k}"] for r in results])
            wt = None
        pooled[k] = (th, wt)

    fig, axes = plt.subplots(1, l96.D_PAR,
                             figsize=(4.0 * l96.D_PAR, 4.0))
    for d, ax in enumerate(axes):
        # frame on the box, widened to cover any unbounded-prior samples
        plo, phi_ = lo[d], hi[d]
        for k in names:
            if k not in BOX_METHODS:
                q = np.percentile(pooled[k][0][:, d], [0.5, 99.5])
                plo, phi_ = min(plo, q[0]), max(phi_, q[1])
        pad = 0.04 * (phi_ - plo)
        plo, phi_ = plo - pad, phi_ + pad
        grid = np.linspace(plo, phi_, 400)
        for k in names:
            th, wt = pooled[k]
            lab, c = METHOD_STYLE[k]
            f = _kde1d(th[:, d], grid, lo[d], hi[d], weights=wt,
                       reflect=(k in BOX_METHODS))
            ax.plot(grid, f, color=c, lw=1.6,
                    label=lab if d == 0 else None)
            # posterior mean: triangle below the axis, one row per
            # method (staggered so overlapping means never occlude)
            mu = (np.average(th[:, d], weights=wt) if wt is not None
                  else th[:, d].mean())
            # INSIDE the axes, near the baseline: below the axis the
            # markers collided first with the tick labels and then with
            # the axis label. A density is high at its own mode, so the
            # strip just above the baseline is free where the mean sits.
            # DF-ETLD's marker on TOP of the stack: METHODS lists it
            # first, so its row is the highest rather than the lowest
            row = len(names) - 1 - names.index(k)
            ax.plot([mu], [0.04 + 0.055 * row], marker="^",
                    ms=7, color=c, clip_on=True, zorder=5,
                    transform=ax.get_xaxis_transform())
        ax.axhline(1.0 / (hi[d] - lo[d]), color="0.6", lw=0.8, ls=":",
                   label="prior" if d == 0 else None)
        ax.axvline(l96.THETA_STAR[l96.PARAM_NAMES[d]],
                   color=style.color("truth"),
                   lw=1.3, ls="--",
                   label=r"true $\theta^*$" if d == 0 else None)
        # the parameter name is a panel IDENTIFIER, not a title:
        # it goes on the axis, as in every other figure here
        # the posterior-mean triangles are stacked BELOW the axis (one
        # row per method), so the label has to clear all of them --
        # otherwise it lands on top of the markers
        ax.set_xlabel(l96.PARAM_LABELS[d])
        ax.set_xlim(plo, phi_)
        ax.set_ylim(bottom=0.0)
        ax.set_yticks([])
    # one y label for the row: the density scale is shared and has no
    # ticks, so repeating it on every panel would only add ink
    axes[0].set_ylabel("generalized posterior")
    # one key for the whole figure, centred beneath it: every entry is
    # shared by all three panels, so it belongs to the figure, not to
    # the first panel where it would crowd the theta1 marginal
    h, lab_ = axes[0].get_legend_handles_labels()
    # neutral swatch for the mean markers: labelling one method's own
    # triangle made the key read as "purple triangle = posterior means",
    # when every method has one in its own colour
    h.append(Line2D([], [], marker="^", ls="none", ms=7, color="0.35"))
    lab_.append("posterior means")
    fig.legend(h, lab_, frameon=False, fontsize=13, loc="lower center",
               ncol=min(len(lab_), 4), bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 1))   # room for the figure legend
    style.save(fig, os.path.join(outdir, f"posterior_{tag}.png"))
    plt.close(fig)


def summarize(rows):
    """
    Aggregate metrics, reported in PRIOR-SD UNITS so parameters with
    different natural ranges are comparable (a raw-units aggregate
    silently weights each parameter by its box width).

      rmse[p]  = sqrt( mean_s (mean_s - theta*_p)^2 )   NATURAL units
      nrmse[p] = sqrt( mean_s (|mean_s - theta*_p| / prior_sd_p)^2 )
      nrmse_all = sqrt( mean over parameters and sets )

    Both are reported: rmse is the quantity to quote per parameter
    (theta1, theta2, sigma_e have interpretable scales), nrmse is the one
    to aggregate ACROSS parameters, since a raw-units aggregate silently
    weights each parameter by its box width.
      cov50 / cov90 = fraction of sets whose equal-tailed credible
                      interval contains theta*_p  (nominal 0.5 / 0.9)
      width90 = mean interval width, also in prior-sd units
    """
    prior_sd = (l96.BOX_HI - l96.BOX_LO) / np.sqrt(12)
    summ = {}
    for k in ALL_METHODS:
        raw = np.array([row[f"err_{k}"] for row in rows])
        e = raw / prior_sd
        s = {"rmse": np.sqrt((raw ** 2).mean(0)).tolist(),
             "nrmse": np.sqrt((e ** 2).mean(0)).tolist(),
             "nrmse_all": float(np.sqrt((e ** 2).mean())),
             "mae": e.mean(0).tolist()}
        for lv in (50, 90):
            c = np.array([row[f"cov{lv}_{k}"] for row in rows], dtype=float)
            w = np.array([row[f"width{lv}_{k}"] for row in rows]) / prior_sd
            s[f"cov{lv}"] = c.mean(0).tolist()
            s[f"width{lv}"] = w.mean(0).tolist()
        summ[k] = s
    return summ


def finalize(results, tag, outdir):
    results = sorted(results, key=lambda r: r["s"])
    make_plots(results, tag, outdir)
    rows = [r["row"] for r in results]
    out = {"tag": tag, "theta_star": l96.THETA_STAR,
           "param_names": l96.PARAM_NAMES,
           "metric_note": ("nrmse/mae/width in prior-sd units; cov50/cov90 "
                           "are equal-tailed credible-interval coverage of "
                           "theta*, nominal 0.5/0.9, over the observation "
                           "sets"),
           "summary": summarize(rows),
           "rows": rows}
    for k in ALL_METHODS:
        out[f"es_{k}_mean"] = np.stack(
            [r["es"][k] for r in results]).mean(0).tolist()
    json.dump(out, open(os.path.join(outdir, f"results_{tag}.json"), "w"),
              indent=1)
    arrays = {}
    for r in results:
        s = r["s"]
        arrays[f"th_dfr_{s}"] = r["th_dfr"]
        arrays[f"th_ad_{s}"] = r["th_ad"]
        arrays[f"th_adk_{s}"] = r["th_adk"]
        arrays[f"th_abc_{s}"] = r["th_abc"]
        arrays[f"w_abc_{s}"] = r["w_abc"]
        arrays[f"th_abcs_{s}"] = r["th_abcs"]
        arrays[f"w_abcs_{s}"] = r["w_abcs"]
        for k, v in r["es"].items():
            arrays[f"es_{k}_{s}"] = v
    np.savez_compressed(os.path.join(outdir, f"samples_{tag}.npz"),
                        **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    tag = "l96"

    n_obs_sets = 2 if args.demo else 5
    t0 = time.time()
    results = []
    for s in range(n_obs_sets):
        r = run_one_set(s, demo=args.demo)
        results.append(r)
        row = r["row"]
        print(f"[set {s}] DF {np.round(row['post_mean_dfr'], 3)}  "
              f"adSGLD-kernel {np.round(row['post_mean_adk'], 3)}  "
              f"ABC {np.round(row['post_mean_abc'], 3)}  "
              f"adSGLD-energy {np.round(row['post_mean_ad'], 3)}  "
              f"| sampler gap {np.round(row['sampler_gap_sd'], 2)} sd  "
              f"({(time.time() - t0) / 60:.1f}m)", flush=True)

    finalize(results, tag, args.outdir)
    print(f"done in {time.time() - t0:.0f}s -> {args.outdir}/")


if __name__ == "__main__":
    main()
