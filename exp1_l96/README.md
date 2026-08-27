# exp1_l96 — well-specified stochastic Lorenz '96 (paper Sec. 3.1, Table 1; appendix energy-score figure)

**This experiment is well-specified.** The observations are generated
from the SAME single-scale model the samplers fit, at
theta* = (theta1, theta2, sigma_e) = (2.0, 0.8, 1.7), by
`l96.generate_observation_ws` — three parameters, with the AR(1)
coefficient held at phi = 0 (`l96.PHI_FIXED`). There is no model error
to recover from, which is the point: it isolates SAMPLER quality from
robustness, so a gap between methods here cannot be blamed on
misspecification.

## Run

```
python run_l96.py --demo     # smoke run, 2 observation sets (minutes)
python run_l96.py            # paper scale: 5 sets (~2.5 h single process)
```

Writes to `outputs/`:

- `results_l96.json` — per-set posterior means, errors, coverage, calls,
  wall-clock, and the aggregate `summary` block. **The paper's Table 1 is
  read from `summary.<method>.rmse` (per parameter, natural units) and
  `summary.<method>.nrmse_all`**, for methods `dfr` (DF-ETLD),
  `adk` (adSGLD-KS), `ad` (adSGLD-ES), `abc` (SMC-ABC raw),
  `abcs` (SMC-ABC summaries).
- `es_per_timestep_l96.png` — **the appendix figure**: posterior-predictive
  energy score per timestep, mean and 95% CI over the five sets.
- `posterior_l96.png` — pooled generalized-posterior marginals (diagnostic;
  not shown in the paper).
- `samples_l96.npz` — raw posterior draws for every method and set, so the
  figures can be restyled without re-running inference.
- `timings_l96.json` — per-method wall-clock measured on one observation
  set at production settings, single process, same hardware. **The paper's
  Table 1 time column comes from this file.**

Each observation set is fully determined by its own rng seed (12000 + s),
so parallelizing over sets is bit-identical to the sequential loop.

## The five methods

- **DF-ETLD** (ours): derivative-free, natural theta space, uniform box
  prior by reflection. M=100 particles, J=10 simulations, 500 steps
  (5e5 calls), w = 30 (beta = w·n on the shared n = 10 series).
- **adSGLD-KS** — the CONTROL: the gradient-based sampler run on
  DF-ETLD's exact kernel-score target (same w, same bandwidth gamma), so
  DF-ETLD vs adSGLD-KS isolates the SAMPLER while adSGLD-KS vs adSGLD-ES
  isolates the SCORING RULE. Exact forward sensitivities through the RK4
  integrator (finite-difference verified); m = 200, 20 000 steps
  (4e6 calls, the benchmark configuration).
- **adSGLD-ES** — same sampler on the energy score.
- **SMC-ABC** (raw Euclidean distance) and **SMC-ABC (6 Hakkarainen
  summaries)** — configuration: 1000 particles × m=10 × 25
  generations = 2.5e5 calls.

## Metrics
- `rmse` — per parameter, natural units. The quantity quoted per
  parameter in Table 1, since theta1, theta2, sigma_e have interpretable
  scales.
- `nrmse`, `nrmse_all` — prior-sd units, for aggregating ACROSS
  parameters (a raw-units aggregate weights each parameter by its box
  width).
- `cov50` / `cov90` — equal-tailed credible-interval coverage of
  theta*, nominal 0.5 / 0.9. Coarse at 5 observation sets: read as a
  direction, not a calibrated number.
- `sampler_gap_sd` — |mean_DF − mean_adSGLD-KS| in posterior-sd units,
  on the shared target: the derivative-free approximation cost, isolated
  from every other design choice.

## Configuration notes

The model is described in full in `l96.py` (general 4-parameter AR(1)
closure; this experiment uses the 3-parameter white-noise version,
phi = 0). The benchmark this configuration follows is cited in
`methods.py`; the scripts themselves are self-contained.
- Simulator: single-scale Lorenz '96 with the Wilks white-noise closure
  g(x) = theta1 + theta2·x + sigma_e·eta, K = 8, F = 10; one RK4 step per
  observation window; noise frozen within a window. See `l96.py` for the
  full discretization conventions.
- Data: 10 series per observation set, 45 recorded states of dimension 8
  (360-dim per series), 5 independent observation sets.
- Prior: uniform box: [1.6, 2.4] × [0.3, 1.3] × [1.2, 2.2]
  (`l96.BOX_LO/BOX_HI`). 
- Bandwidth: median heuristic over prior-predictive simulations,
  computed once per observation set and frozen (`common/mmd.py`).
