"""
model.py
========
The nonlinearly-perturbed location model and all closed-form quantities.

Forward map (scalar theta, scalar data):

    G_theta(u) = g(theta) + u,      g(theta) = theta + alpha * theta^3
    (default cubic map; see MAP_KIND below for the quadratic variant),
    u ~ N(0, sigma^2)                                            (M1)

so that  P_theta = N(g(theta), sigma^2).

Kernel: Gaussian,  k(x, y) = exp( -(x - y)^2 / (2 gamma^2) ).     (M2)

Why this model
--------------
1) Every expectation appearing in MMD^2(P_theta, P_hat_data) is available in
   closed form (see below), so the *exact* MMD-Bayes posterior can be
   evaluated on a grid with zero Monte Carlo error  -> reference.py.
2) alpha = 0 makes G affine, the regime where the statistical linearization
   of the paper (Proposition 1) is exact -> unit test.
3) g''(theta) = 2 alpha everywhere, so the Taylor-remainder heuristic
   predicts a drift bias ~ alpha * tr(C_s).

Closed forms used for the reference posterior
---------------------------------------------
For x, x' ~ P_theta i.i.d. we have x - x' ~ N(0, 2 sigma^2), and the standard
Gaussian integral  E[exp(-z^2/(2 gamma^2))] = gamma / sqrt(gamma^2 + s^2)
for z ~ N(mu, s^2) generalizes to

    E exp(-z^2/(2 gamma^2)) = gamma/sqrt(gamma^2+s^2)
                              * exp( - mu^2 / (2 (gamma^2 + s^2)) ).   (M3)

Hence:
    E_{x,x'~P_theta} k(x, x') = gamma / sqrt(gamma^2 + 2 sigma^2)      (M4)
    E_{x~P_theta} k(x, y^n)   = gamma / sqrt(gamma^2 + sigma^2)
                * exp( -(g(theta)-y^n)^2 / (2 (gamma^2+sigma^2)) )     (M5)

With the empirical data measure P_hat = (1/N) sum_n delta_{y^n},

    MMD^2(P_theta, P_hat) = (M4)
                          + (1/N^2) sum_{n,n'} k(y^n, y^{n'})   [const in theta]
                          - (2/N) sum_n (M5).                          (M6)

The data-data term is constant in theta and therefore cancels in the
normalized posterior; we still compute it so that reported MMD^2 values are
absolute, comparable across configurations, and >= 0 up to estimator choice.

Exact gradient (used by the ALDI-exact-gradient rung of the ladder):

    d/dtheta MMD^2 = (2 g'(theta) / (gamma^2 + sigma^2)) *
                     (1/N) sum_n (g(theta) - y^n) * (M5)_n              (M7)

obtained by differentiating (M5) through the chain rule
(g'(theta) = 1 + 2 alpha theta).
"""

import numpy as np


# ----------------------------------------------------------------------
# forward map
# ----------------------------------------------------------------------
# Two nonlinear mean maps are provided:
#
#   "cubic"     g(theta) = theta + alpha * theta^3   (DEFAULT)
#               strictly monotone for alpha >= 0 (g' = 1 + 3 alpha theta^2
#               > 0), hence injective: the reference posterior is UNIMODAL
#               for every alpha, and W1-to-reference measures linearization
#               bias only -- never mode-missing.
#
#   "quadratic" g(theta) = theta + alpha * theta^2
#               non-injective with a fold at theta = -1/(2 alpha); the
#               mirror solution of g(theta) = g(theta_star) is
#               theta = -1/alpha - theta_star, which for moderate alpha
#               lies well inside the prior -> genuinely BIMODAL reference.
#               Validation runs confirmed this becomes material already at
#               alpha ~ 0.25-1 with the default prior, contaminating the
#               bias measurement with mode-missing error. Kept here because
#               it is exactly the free bimodal testbed for the later
#               multimodality/tempering section -- just not for THIS study.
#
# In both cases alpha = 0 gives the affine map (linearization exact), and
# only g(theta) enters the closed forms (M4)-(M7), so nothing else changes.
MAP_KIND = "cubic"


def g(theta, alpha):
    """Mean map, see MAP_KIND above.  (M1)"""
    theta = np.asarray(theta, dtype=float)
    if MAP_KIND == "cubic":
        return theta + alpha * theta ** 3
    return theta + alpha * theta ** 2


def g_prime(theta, alpha):
    """g'(theta), needed for the exact gradient (M7)."""
    theta = np.asarray(theta, dtype=float)
    if MAP_KIND == "cubic":
        return 1.0 + 3.0 * alpha * theta ** 2
    return 1.0 + 2.0 * alpha * theta


def simulate(theta, alpha, sigma, U):
    """
    Push latent noise through the generator: x^{mj} = g(theta^m) + u^j.

    Parameters
    ----------
    theta : (M,) particle values
    U     : (J,) latent draws u^j ~ N(0,1) *shared across particles*
            (common random numbers, as in the paper's Section 4 setup:
            the same {u^j} are transformed by every G_{theta^m}).

    Returns
    -------
    X : (M, J) with X[m, j] = g(theta^m) + sigma * u^j
    """
    return g(theta, alpha)[:, None] + sigma * U[None, :]


def generate_data(theta_star, alpha, sigma, n_obs, rng):
    """Observed data y^n ~ P_{theta_star}, n = 1..N (well-specified case)."""
    return g(theta_star, alpha) + sigma * rng.standard_normal(n_obs)


# ----------------------------------------------------------------------
# kernel and bandwidth
# ----------------------------------------------------------------------
def gaussian_kernel(x, y, gamma):
    """k(x,y), broadcasting over arrays.  (M2)"""
    return np.exp(-((x - y) ** 2) / (2.0 * gamma ** 2))


def grad_x_gaussian_kernel(x, y, gamma):
    """
    d/dx k(x, y) = -(x - y)/gamma^2 * k(x, y).
    This is the *kernel* gradient: analytic and cheap. Note it is exactly the
    derivative DF-ETLD keeps -- the method is simulator-derivative-free
    (Jacobian-free), not free of all derivatives.
    """
    return -((x - y) / gamma ** 2) * gaussian_kernel(x, y, gamma)


def median_heuristic(y):
    """
    Bandwidth gamma = median of pairwise distances |y_i - y_j|, i < j,
    computed ONCE on the observed data and then frozen for every method and
    every alpha (design decision: bandwidth must not confound the bias study).
    The particular convention (median vs median/sqrt(2)) is immaterial as long
    as it is used consistently.
    """
    diffs = np.abs(y[:, None] - y[None, :])
    iu = np.triu_indices(len(y), k=1)
    return float(np.median(diffs[iu]))


# ----------------------------------------------------------------------
# closed-form MMD^2 against the empirical data measure, and its gradient
# ----------------------------------------------------------------------
def mmd2_exact(theta, y, alpha, sigma, gamma):
    """
    Exact population-vs-empirical MMD^2(P_theta, P_hat_data), eq. (M6).
    Vectorized over an array of theta values (used on the reference grid and
    by the ALDI-exact-gradient sampler).
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    N = len(y)

    # model-model term (M4): constant in theta but kept for absolute values
    t_xx = gamma / np.sqrt(gamma ** 2 + 2.0 * sigma ** 2)

    # data-data term: (1/N^2) sum_{n,n'} k(y^n, y^{n'}) -- constant in theta.
    # (The paper's empirical estimator (6) uses the unbiased off-diagonal
    #  version; the difference is a theta-independent constant and cancels in
    #  the posterior, so the V-statistic here is deliberate and harmless.)
    t_yy = float(np.mean(gaussian_kernel(y[:, None], y[None, :], gamma)))

    # cross term (M5), averaged over data
    c = gamma / np.sqrt(gamma ** 2 + sigma ** 2)
    resid = g(theta, alpha)[:, None] - y[None, :]              # (T, N)
    e_kxy = c * np.exp(-resid ** 2 / (2.0 * (gamma ** 2 + sigma ** 2)))
    t_xy = e_kxy.mean(axis=1)                                   # (T,)

    return t_xx + t_yy - 2.0 * t_xy


def grad_mmd2_exact(theta, y, alpha, sigma, gamma):
    """
    Exact d/dtheta MMD^2, eq. (M7). This is what the gradient-based ALDI rung
    uses, so that rung 2 of the ladder targets the same posterior as rung 1
    with NO linearization anywhere -- (rung 3 - rung 2) then isolates the
    statistical-linearization bias by construction.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    c = gamma / np.sqrt(gamma ** 2 + sigma ** 2)
    v = gamma ** 2 + sigma ** 2
    resid = g(theta, alpha)[:, None] - y[None, :]              # (T, N)
    e_kxy = c * np.exp(-resid ** 2 / (2.0 * v))
    # d/dtheta [ -(2/N) sum_n E k(x, y^n) ]  = (2 g'/v) * mean_n resid * E k
    return (2.0 * g_prime(theta, alpha) / v) * np.mean(resid * e_kxy, axis=1)


# ----------------------------------------------------------------------
# prior
# ----------------------------------------------------------------------
def log_prior(theta, m0, s0):
    """log N(theta | m0, s0^2), up to an additive constant."""
    return -0.5 * ((np.asarray(theta) - m0) / s0) ** 2


def grad_log_prior(theta, m0, s0):
    """d/dtheta log prior = -(theta - m0)/s0^2."""
    return -(np.asarray(theta) - m0) / s0 ** 2
