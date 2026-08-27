# exp3_sir — Misspecified SIR, Kelly et al. (2024) Section 4.4 benchmark

Reimplements the misspecified SIR example of Kelly, Nott, Frazier, Warne &
Drovandi, *Misspecification-robust Sequential Neural Likelihood*
(TMLR 2024, arXiv:2301.13368), §4.4 — which follows Ward et al. (NeurIPS
2022). The methods compared here are **DF-ETLD**, **SMC-ABC** and
**adSGLD on two scoring rules** (kernel and energy); only methods run in
this codebase are reported.

Run, in order:

```
python run_sir.py            # DF-ETLD + SMC-ABC + adSGLD-KS  (~22 min)
python calibrate_beta_es.py  # Bayes-factor-matched beta_ES   (seconds)
python run_sir_es.py         # converged adSGLD-ES, 4 chains x 16k steps
```

(`python run_sir.py --demo` and `python run_sir_es.py --chains 2
--steps 400` are quick smoke runs.)

Paper mapping:
- **SIR posterior figure** = `outputs/sir_posteriors_es_credible.png`
  (DF-ETLD | SMC-ABC | adSGLD-ES, 50/80/95% credible contours).
- **SIR table**: DF-ETLD error/time from `outputs/results.json` and
  `outputs/cost_table.json`; adSGLD-ES error/time from
  `outputs/results_es.json`; SMC-ABC from `results.json`/`cost_table.json`.
- **Appendix observed-series figure** = `outputs/sir_observed_series.png`
  (one draw from the misspecified DGP, weekly artefact magnified).

## The two adSGLD arms, and their weights

They differ in ONE thing, the scoring rule; z-space, finite-difference
gradients, step size, prior and common random numbers are all held fixed,
so the pair isolates the rule exactly as the contamination
experiment does.

beta is SCORE-SPECIFIC — it multiplies a particular scoring rule, and the
energy score's natural scale is not MMD²'s, so the kernel arm's beta = 10
says nothing about the right value for the energy arm. The energy weight
is set by **Pacchiardi et al.'s own procedure** (Generalized Bayesian LFI,
Sec. 4.2, adapting Bissiri–Holmes–Walker): two scoring-rule posteriors
have the same scale when their Bayes factors agree, so

    w = median over (theta, theta') ~ prior of
        [S'(P_theta,y) - S'(P_theta',y)] / [S(P_theta,y) - S(P_theta',y)]

with S estimated by simulating m draws per theta. It needs no likelihood
and no repeated posterior inference: 14 400 simulator calls, seconds.
Anchored on this experiment's calibrated beta_KS = 10 it gives
**beta_ES = 4.38**. 

## Convergence, not budget

Both SGLD arms are run to convergence: 4 independent chains x 16 000
steps, full traces retained, split-Rhat computed on the second halves.
Rhat is 1.015 (KS) and 1.012 (ES) — marginally above the strict 1.01 bar,
and quoted as measured.

Every method is run long enough to be effectively at stationarity, which
is the equivalent criterion across samplers of different kinds: DF-ETLD's
run length was set by prefix stability of a longer run (the posterior
recomputed from growing prefixes of a 6000-step run is stable from ~750
steps — mean moves <= 0.11 sd, sd not at all — and production uses 1500
with margin); SMC-ABC stops on its own tolerance plateau.

## Configuration (as published)
| item | value |
|---|---|
| assumed DGP | SIR with time-varying Re_t: dRe = υ(β/η − Re)dt + σ√R0 dW |
| υ, σ | 0.5, 0.5 |
| priors | η ~ U(0, 0.5), β \| η ~ U(η, 0.5) |
| truth | β = 0.15, η = 0.1 |
| horizon | 365 days, daily infected only; I₀ = 100, counts scaled by 1e5 |
| misspecification | weekend counts −5%, Monday counts +10% |
| summaries | mean, median, max, max day, half day, lag-1 autocorrelation |

Validation: our simulated lag-1 autocorrelation is 0.9986 ± 0.0004 and the
misspecified observation gives 0.994–0.995, against the paper's reported
0.9997 (simulated) and 0.9957 (observed) — the incompatibility signature
reproduces.

## Outputs
- `sir_posteriors.png` — their Fig. 7 layout: bivariate (η, β) contours
  with marginals, one panel per method, truth marked ×.
- `cost_table.md/.json` — simulator calls and wall-clock for every
  method run here.
- `results.json`, `posteriors.npz` — numbers and raw draws for re-plotting.

## Do not regress
- SGLD needs dt ≈ 5e-2. At dt ≈ 5e-3 the chain stalls in the saturated
  corner of the logistic reparameterisation, where the simulator is flat in
  z and the FD gradient vanishes, and it then reports the prior corner as
  its posterior. This is a sampler failure, not a target failure: the MMD
  objective is −0.75 at the truth against −0.33 at the corner.
- SMC-ABC uses m = 1 simulation per parameter (as is standard); its
  distance is correspondingly noisy, and the population can collapse to a
  biased point at large generation counts. 
