# exp2_contamination — misspecified location model (paper Sec. 3.2, Fig. 1; appendix error-vs-eps figure)

A deliberately simple model in which the effect of misspecification is
transparent. Observations are drawn from a Huber contamination,

    y_i ~ (1 - eps) N(theta*, 1) + eps N(z, 1),      theta* = 1,

with `eps` the fraction of contaminated observations and `z` the outlier
location. N = 100 observations, prior theta ~ N(2, 1), 10 independent
seeds per configuration. The outlier count is exact (`round(eps*N)`), so
panels differ by the contamination itself and not by a random number of
outliers.

Because standard Bayes is conjugate here, its posterior is available in
closed form and carries no sampler error: its curve *is* the
contamination bias, which is what makes this benchmark a clean read on
robustness rather than on sampling.

## Run

Run the grid first — it calibrates the learning rates that the error
figure then reads.

```
python run_posterior_grid.py    # ~25 min  -> contamination_posterior_grid.png
python run_contamination.py     # ~25 min  -> contamination.png
python replot_figures.py        # seconds: redraw both from committed results
```

Both runners take `--demo` for a fast smoke run and `--outdir` to write
elsewhere.

Writes to `outputs/`:

- `contamination_posterior_grid.png` — **the paper's Figure 1**: posterior
  shapes on a 3x3 grid (row 1 varies eps in {0, 0.1, 0.2} at z = 10;
  row 2 varies z in {1, 10, 20} at eps = 0.1; columns are standard Bayes,
  DF-ETLD, SGLD-ES).
- `contamination_posterior_grid_samples.npz`, `..._config.json` — the
  pooled draws behind each panel, and the calibrated learning rates
  (`w_kernel`, `w_energy`) plus the bandwidth and target sd.
- `contamination.png`, `contamination.json` — **the appendix figure**:
  posterior-mean error against eps in {0, 0.1, 0.2, 0.3, 0.4}, four
  methods, with the contamination-bias reference line.

`replot_figures.py` redraws both figures from these committed files
without re-running inference — use it after any styling change.

## The four methods, and why this particular four

- **standard Bayes** — conjugate, closed form, so it carries no sampler
  error and its curve is the contamination bias itself.
- **DF-ETLD** (ours) — kernel score, derivative-free.
- **SGLD-KS** — the same posterior as DF-ETLD (same w, same bandwidth)
  reached with gradients. DF-ETLD against this isolates the **sampler**.
- **SGLD-ES** — the same sampler on the energy score. Against SGLD-KS it
  isolates the **scoring rule**.

So the figure separates "is the derivative-free sampler paying a price"
from "is the kernel score buying robustness", which a three-method
version cannot do.

## Why plain SGLD rather than adSGLD

The gradient-based arms here are unpreconditioned, deliberately. With
D = 1 an adaptive diagonal preconditioner has no anisotropy to correct,
so it reduces to a step-size rescaling — and RMSprop's rescaling
normalises the drift to ~step/2 whatever the gradient scale, which stops
the chain reaching the mode and makes the w-calibration chase a
non-stationary spread. The gradients here are also exact and analytic
(for x = theta + sigma*u we have dx/dtheta = 1, and both scores'
model-model terms are theta-independent, so those gradients vanish), so
there is no gradient noise for adaptivity to manage either. Hence SGLD,
not adSGLD. The Lorenz '96 arm in `exp1_l96` **is** RMSprop-preconditioned
and keeps the adSGLD name.

## Conventions that make the comparison fair

- **Learning rates are calibrated, not chosen.** The three posteriors are
  different objects whose raw concentrations are not comparable, and a
  method looks spuriously robust if its posterior is merely wider. Each
  generalized posterior's `w` is therefore calibrated **once on clean
  data** (eps = 0) so its posterior sd matches standard Bayes there, then
  **frozen** for every panel. Differences across panels are then
  robustness, not width. `run_contamination.py` reads those rates from
  `contamination_posterior_grid_config.json` rather than re-fitting them,
  so the two figures describe one experiment.
- **Bandwidth: median heuristic over simulator draws, frozen.** See `model_gamma`.
- **The sweep stops at eps = 0.5** because the breakdown point of a
  translation-equivariant estimator is 1/2; beyond it the contamination
  is the data and the estimand itself has moved.
- Every panel pools 10 independent datasets, so the curves carry
  within-dataset uncertainty and dataset-to-dataset variability together.
