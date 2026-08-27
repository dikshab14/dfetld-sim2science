"""
config.py
=========
Single source of truth for all experiment settings.

Every parameter that appears in the experimental design of the
nonlinearity-bias study lives here, so that a sweep is expressed as
"copy the default config, change one field" -- which is what guarantees
that the comparison ladder (exact grid / ALDI-exact-gradient / DF-ETLD)
shares every setting except the one under study.

Notation matches the paper (Appendix, empirical linearization study):
    theta  : scalar parameter (D = 1)
    G_theta(u) = g(theta) + u,  g(theta) = theta + alpha * theta**3
                 (the default monotone cubic map; see model.MAP_KIND)
    u ~ N(0, sigma^2)
    prior: theta ~ N(m0, s0^2)
    MMD kernel: Gaussian, k(x,y) = exp(-(x-y)^2 / (2 gamma^2))
    generalized posterior:  pi(theta) prop. prior(theta) * exp(-beta * MMD^2)
"""

from dataclasses import dataclass, field, replace


@dataclass
class Config:
    # ---------------- model / data generating process ----------------
    alpha: float = 0.25        # nonlinearity dial; alpha = 0 -> linear map,
                               # where statistical linearization is EXACT
                               # (this is the built-in unit test of the study)
    sigma: float = 1.0         # observation noise std of the forward map
    theta_star: float = 0.5    # true parameter used to generate data.
    n_obs: int = 100           # N: number of observations y^1..y^N

    # ---------------- generalized-Bayes ingredients -------------------
    beta: float = 100.0        # inverse temperature in exp(-beta * MMD^2).
                               # MMD^2 is O(1) (it is an average, not a sum),
                               # so beta must scale like N to give the data
                               # weight comparable to a standard posterior;
                               # default beta = N = 100.
    m0: float = 0.0            # prior mean
    s0: float = 2.0            # prior std (diffuse relative to posterior)

    # gamma (kernel bandwidth) is NOT set here: it is computed ONCE from the
    # observed data via the median heuristic (model.median_heuristic) and then
    # held fixed across methods and across alpha, so bandwidth adaptation can
    # never act as a confounder of the bias measurement.

    # ---------------- initialization ----------------------------------
    init_mode: str = "reference"
    # "reference": warm-start BOTH samplers from i.i.d. draws of the exact
    #   grid posterior (inverse-CDF sampling). This isolates STATIONARY bias:
    #   a sampler with the correct invariant measure stays at the
    #   finite-sample floor; a biased one drifts to its own stationary state,
    #   and the measured W1 is that bias.
    # "prior": particles from the prior, as in the paper -- the practical
    #   variant, reported in an appendix with a W1-vs-time trace.

    # ---------------- sampler settings (shared by both rungs) ---------
    M: int = 20                # ensemble size (number of particles)
    J: int = 50                # simulations per particle per step (DF-ETLD)
    dt: float = 1e-2           # Euler-Maruyama step size Delta s
    n_steps: int = 30400       # total time steps
    burn_in: int = 400         # discarded steps (spin-up)
    thin: int = 10             # keep every `thin`-th post-spin-up iterate

    # ---------------- reference posterior (grid) ----------------------
    grid_lo: float = -14.0     # grid support for exact posterior. Must cover
    grid_hi: float = 14.0      # the PRIOR (m0 +/- 7 s0), not just typical
                               # posteriors: at small beta the posterior is
                               # nearly the prior, and reference.py raises if
                               # mass touches the boundary.
    grid_size: int = 28001     # fine enough that quadrature error ~ machine eps

    # ---------------- experiment bookkeeping --------------------------
    n_seeds: int = 10          # R: independent repetitions per configuration.
                               # Figures plot mean +/- SEM = std/sqrt(R), so
                               # the bars scale as 1/sqrt(R): R = 10 narrows
                               # them ~1.6x relative to R = 4, and R = 20 would
                               # gain a further 1.4x. Runtime is linear in R
                               # (~5 min for Figure A at R = 10).
    seed0: int = 12345         # base seed; run r uses seed0 + r


# Convenience: derive a modified copy (used everywhere in sweeps)
def with_(cfg: Config, **kwargs) -> Config:
    """Return a copy of cfg with the given fields replaced."""
    return replace(cfg, **kwargs)


# ---------------- sweep grids used in the paper figures ----------------
# Figure A: bias vs nonlinearity (headline figure)
ALPHA_SWEEP = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]

# Figure B: bias vs ensemble size, at three nonlinearity levels
M_SWEEP = [5, 10, 20, 50]
ALPHA_FOR_M_SWEEP = [0.0, 0.25, 1.0]


