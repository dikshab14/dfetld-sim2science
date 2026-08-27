"""
exp3_sir/sir_model.py
=========================
The misspecified SIR benchmark as specified in Ward et al. (NeurIPS 2022).

ASSUMED DGP (the simulator we infer with)
-----------------------------------------
Compartmental SIR with infection rate beta and recovery rate eta:

    dS/dt = -beta~_t S I,   dI/dt = beta~_t S I - eta I,   dR/dt = eta I

extended stochastically through a time-varying effective reproduction
number Re_t = beta~_t / eta obeying an Ornstein-Uhlenbeck-type SDE:

    dRe_t = upsilon (beta/eta - Re_t) dt + sigma sqrt(R0) dW_t

with upsilon = 0.5 (reversion rate), sigma = 0.5 (volatility), W_t standard
Brownian motion, and R0 = beta/eta the long-run mean. The instantaneous
rate is recovered as beta~_t = eta * Re_t.

Published configuration:
  * priors      eta ~ U(0, 0.5),  beta | eta ~ U(eta, 0.5)
  * truth       beta = 0.15, eta = 0.1   (so R0 = 1.5)
  * horizon     365 days; ONLY the daily number of infected is observed
  * initial     100 infected; counts scaled by 100,000 (i.e. the state is
                a fraction, reported as fraction * 1e5)
  * summaries   six: mean, median, max, day of max, day at which half the
                total infections was reached, lag-1 autocorrelation

TRUE DGP (reality; the misspecification)
----------------------------------------
A reporting lag: weekend counts are reduced by 5%, and Monday counts are
increased by 10%. The assumed DGP has no mechanism producing this
weekly sawtooth, so the observed lag-1 autocorrelation cannot be matched:
the paper reports observed autocorrelation 0.9957 against simulated values
tightly centred on 0.9997. 

Reparameterisation for gradient/Langevin methods
------------------------------------------------
The prior support is the triangle {0 < eta < 0.5, eta < beta < 0.5}.
Mapping to unconstrained z:

    eta  = 0.5 * s(z1),      beta = eta + (0.5 - eta) * s(z2),  s = sigmoid

the Jacobian is triangular with determinant 0.5 s'(z1) (0.5 - eta) s'(z2),
and the uniform prior on the triangle has density 1 / (0.5 (0.5 - eta)),
so the induced prior on z is EXACTLY a product of standard logistics:

    p(z) = s'(z1) s'(z2),    grad log p(z_i) = 1 - 2 s(z_i).

No clipping, no rejection, and every method shares the identical prior
measure.
"""

import numpy as np

# ---- published configuration -------------------------------------------
UPSILON, SIGMA_OU = 0.5, 0.5
T_DAYS = 365
DT = 0.1                      # EM step, days (10 substeps per day)
N_SUB = int(round(1.0 / DT))
I0_COUNT = 100.0
SCALE = 1e5                   # counts reported as fraction * SCALE
BETA_TRUE, ETA_TRUE = 0.15, 0.1
THETA_TRUE = np.array([BETA_TRUE, ETA_TRUE])
WEEKEND_REDUCE, MONDAY_INCREASE = 0.05, 0.10
SUMMARY_NAMES = ["mean", "median", "max", "max day", "half day", "autocorr"]


def simulate_prevalence(theta, U):
    """
    Vectorised Euler-Maruyama simulation of the assumed DGP.

    theta : (M, 2) rows (beta, eta) in NATURAL space
    U     : (J, T_DAYS * N_SUB) standard normals driving W_t, SHARED across
            particles (common random numbers)
    returns X : (M, J, T_DAYS) daily infected counts (fraction * SCALE)
    """
    theta = np.atleast_2d(theta)
    M, J = theta.shape[0], U.shape[0]
    beta = theta[:, 0][:, None]
    eta = np.maximum(theta[:, 1][:, None], 1e-8)
    R0 = beta / eta                                    # long-run mean

    S = np.full((M, J), 1.0 - I0_COUNT / SCALE)
    I = np.full((M, J), I0_COUNT / SCALE)
    Re = np.broadcast_to(R0, (M, J)).copy()            # start at the mean
    out = np.empty((M, J, T_DAYS))
    sq = np.sqrt(DT)
    k = 0
    for d in range(T_DAYS):
        for _ in range(N_SUB):
            dW = U[None, :, k] * sq
            Re = Re + UPSILON * (R0 - Re) * DT + SIGMA_OU * np.sqrt(R0) * dW
            Re = np.maximum(Re, 0.0)                   # rate stays >= 0
            b_t = eta * Re                             # beta~_t
            new_inf = b_t * S * I * DT
            rec = eta * I * DT
            new_inf = np.clip(new_inf, 0.0, S)
            rec = np.clip(rec, 0.0, I + new_inf)
            S = S - new_inf
            I = I + new_inf - rec
            k += 1
        out[:, :, d] = I * SCALE                       # observed prevalence
    return out


def draw_latents(rng, J):
    """Brownian increments (standard normals) for one simulator call."""
    return rng.standard_normal((J, T_DAYS * N_SUB))


def apply_reporting_lag(x):
    """
    The misspecification: weekend counts reduced by 5%, Monday counts
    increased by 10%. Day 0 is taken to be a Monday, so weekdays are
    d % 7 in {5, 6} (Sat, Sun) and Mondays are d % 7 == 0.
    Operates on the last axis (days).
    """
    x = np.array(x, dtype=float, copy=True)
    d = np.arange(x.shape[-1])
    x[..., (d % 7 == 5) | (d % 7 == 6)] *= (1.0 - WEEKEND_REDUCE)
    x[..., d % 7 == 0] *= (1.0 + MONDAY_INCREASE)
    return x


def summaries(x):
    """
    The six published summaries: mean, median, max, day of max, day at
    which half the total infections was reached, lag-1 autocorrelation.
    x : (..., T_DAYS) -> (..., 6)
    """
    x = np.asarray(x, dtype=float)
    mean = x.mean(-1)
    med = np.median(x, -1)
    mx = x.max(-1)
    max_day = np.argmax(x, -1).astype(float)
    cum = np.cumsum(x, -1)
    half_day = (cum < 0.5 * cum[..., -1:]).sum(-1).astype(float)
    c = x - x.mean(-1, keepdims=True)
    num = (c[..., 1:] * c[..., :-1]).sum(-1)
    den = np.sqrt((c[..., 1:] ** 2).sum(-1) * (c[..., :-1] ** 2).sum(-1))
    ac1 = num / np.maximum(den, 1e-12)
    return np.stack([mean, med, mx, max_day, half_day, ac1], -1)


def make_observation(rng, misspecified=True):
    """One observed 365-day series at the true parameters."""
    x = simulate_prevalence(THETA_TRUE[None], draw_latents(rng, 1))[0, 0]
    return apply_reporting_lag(x) if misspecified else x


# ---- unconstrained reparameterisation of the triangular prior ----------
def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def z_to_theta(z):
    """z (…, 2) -> theta (…, 2) = (beta, eta) inside the prior triangle."""
    z = np.atleast_2d(z)
    eta = 0.5 * _sig(z[:, 0])
    beta = eta + (0.5 - eta) * _sig(z[:, 1])
    return np.stack([beta, eta], -1)


def sample_z_prior(rng, size):
    """Uniform on the triangle == i.i.d. standard logistic in z."""
    u = rng.uniform(size=(size, 2))
    return np.log(u / (1.0 - u))


def grad_log_prior_z(z):
    """d/dz log p(z) for the standard logistic: 1 - 2 sigmoid(z)."""
    return 1.0 - 2.0 * _sig(z)


# ---- alternative prior: UNIFORM on the triangle ------------------------
# beta ~ U(0, 0.5), gamma ~ U(0, 0.5), rejecting gamma > beta. This is NOT
# the same as the published eta ~ U(0,0.5), beta|eta ~ U(eta,0.5): the
# conditional construction has density 2/(0.5 - eta), which diverges as
# eta -> 0.5 and so tilts towards large recovery rates, whereas rejection
# sampling from the square is uniform on the triangle. Implied R0 differs
# accordingly (median 1.40 vs 2.00; P(R0 < 1.1) = 0.24 vs 0.09).

def grad_log_prior_z_uniform(z):
    """d/dz log p(z) for the uniform-on-triangle prior."""
    z = np.atleast_2d(z)
    return np.stack([1.0 - 3.0 * _sig(z[:, 0]),
                     1.0 - 2.0 * _sig(z[:, 1])], -1)


def sample_theta_prior_uniform(rng, size):
    """beta, gamma ~ U(0,0.5) with gamma > beta rejected (acceptance 1/2)."""
    out = np.empty((size, 2))
    n = 0
    while n < size:
        b = rng.uniform(0.0, 0.5, size)
        g = rng.uniform(0.0, 0.5, size)
        ok = g < b
        take = min(size - n, int(ok.sum()))
        out[n:n + take] = np.stack([b[ok][:take], g[ok][:take]], -1)
        n += take
    return out


def theta_to_z(theta):
    """Inverse of z_to_theta, for initialising from theta-space draws."""
    theta = np.atleast_2d(theta)
    beta, eta = theta[:, 0], theta[:, 1]
    u1 = np.clip(eta / 0.5, 1e-9, 1 - 1e-9)
    u2 = np.clip((beta - eta) / (0.5 - eta), 1e-9, 1 - 1e-9)
    return np.stack([np.log(u1 / (1 - u1)), np.log(u2 / (1 - u2))], -1)


def sample_z_prior_uniform(rng, size):
    """Draws from the uniform-on-triangle prior, expressed in z."""
    return theta_to_z(sample_theta_prior_uniform(rng, size))


def sample_theta_prior(rng, size):
    """Direct draws from eta ~ U(0,0.5), beta | eta ~ U(eta,0.5)."""
    eta = rng.uniform(0.0, 0.5, size)
    beta = rng.uniform(eta, 0.5)
    return np.stack([beta, eta], -1)
