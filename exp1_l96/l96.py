"""
exp1_l96/l96.py
===============
Stochastic Lorenz '96 simulator and experiment configuration.

MODEL. The single-scale Lorenz '96 system with a stochastic closure for
the unresolved tendency (Wilks 2005):

    dx_k/dt = -x_{k-1}(x_{k-2} - x_{k+1}) - x_k + F - g(x_k) + eta(t),
    g(x) = theta1 + theta2 * x,

with K = 8 sites, forcing F = 10, cyclic indices. In its GENERAL form
the closure noise eta follows an AR(1) process per integration window,

    eta_w = phi * eta_{w-1} + sigma_e * sqrt(1 - phi^2) * N(0, 1),

which gives a 4-PARAMETER model, theta = (theta1, theta2, sigma_e, phi).

THIS EXPERIMENT USES THE 3-PARAMETER WHITE-NOISE VERSION: phi is fixed
at 0 (PHI_FIXED below), so eta is i.i.d. N(0, sigma_e^2) per window and
the inferred parameter is theta = (theta1, theta2, sigma_e) -- the model
described in the paper's appendix. The simulator implements the general
AR(1) form (its internals are 4-wide) and appends the fixed phi column
automatically; the 4-parameter AR(1) variant is a strict generalization
and is NOT used in the paper.

WELL-SPECIFIED BY CONSTRUCTION: observations are generated from this
same 3-parameter model at theta* = (2, 0.8, 1.7), so there is no model
error and any gap between methods is the sampler, not robustness.

DISCRETIZATION: time grid linspace(0, 1.5, 45) -> 44 windows of
dt = 1.5/44 ~ 0.0341. One classical RK4 step per window; the
deterministic closure g(x) is evaluated at every RK4 stage, only the
noise eta is frozen within the window. eta is initialized as
sigma_e sqrt(1-phi^2) N(0,1) (the non-stationary initialization).

DATA: the recorded series INCLUDES the initial state as its first
column: dim = 45 x 8 = 360 per series. (The IC column is identical in
simulations and observations, so it contributes nothing to any
distance.) One observation set = n = 10 series simulated at theta* from
the fixed, KNOWN initial state X0_FIXED -- the initial condition is not
inferred.

PRIOR: uniform on the box [1.6, 2.4] x [0.3, 1.3] x [1.2, 2.2].

The benchmark this configuration follows is cited in methods.py.
"""

import numpy as np

# ---- model configuration -------------------------------------------------
K_SLOW = 8
F_F = 10.0
T_UNITS = 1.5
N_REC = 45                       # recorded points incl. the IC column
N_WIN = N_REC - 1                # 44 RK4 windows
DT_WIN = T_UNITS / N_WIN         # ~0.0341

# fixed initial state, shared by observations and simulations (the IC is
# known, not inferred)
X0_FIXED = np.array([6.4558, 1.1054, -1.4502, -0.1985,
                     1.1905, 2.3887, 5.6689, 6.7284])

# ---- prior box and true parameter ----------------------------------------
# phi fixed at 0: the white-noise closure (see module header).
PHI_FIXED = 0.0
# Uniform box prior: the benchmark's interval widths (0.8, 1.0, 1.0),
# each re-centred on theta* so that no parameter carries an off-centre
# no-information floor in its recovery error (see README).
BOX_LO = np.array([1.6, 0.3, 1.2])
BOX_HI = np.array([2.4, 1.3, 2.2])
PARAM_NAMES = ["theta1", "theta2", "sigma_e"]
# what FIGURES show: the machine keys above index JSON and npz files and
# must stay ASCII, but an axis should read as mathematics
PARAM_LABELS = [r"$\theta_1$", r"$\theta_2$", r"$\sigma_e$"]
THETA_STAR = {"theta1": 2.0, "theta2": 0.8, "sigma_e": 1.7}
D_PAR = 3


def n_latents():
    """(steps, K) of latent normals per series: 1 init + N_WIN updates."""
    return N_REC, K_SLOW


def _adv(x):
    """Advection term -x_{k-1}(x_{k-2} - x_{k+1}), cyclic, last axis."""
    return -np.roll(x, 1, -1) * (np.roll(x, 2, -1) - np.roll(x, -1, -1))


def simulate(theta, x0, U):
    """
    Vectorized simulator (general 4-parameter AR(1) form; in 3-parameter
    mode the fixed phi column is appended automatically).

    theta : (M, 4) parameters (theta1, theta2, sigma_e, phi) in the box
    x0    : (K,) initial state
    U     : (Jsim, N_REC, K) standard normals, SHARED across particles
    returns X : (M, Jsim, N_REC * K)  -- first K entries are the IC column

    Per window: one RK4 step of  adv(x) - x + F - (theta1 + theta2 x) +
    eta,  with eta frozen; then the AR(1) update of eta.
    """
    theta = np.atleast_2d(theta)
    if theta.shape[1] == 3:                       # 3-param mode: phi fixed
        theta = np.column_stack(
            [theta, np.full(theta.shape[0], PHI_FIXED)])
    M, Jsim = theta.shape[0], U.shape[0]
    th1 = theta[:, 0][:, None, None]
    th2 = theta[:, 1][:, None, None]
    sig = theta[:, 2][:, None, None]
    phi = theta[:, 3][:, None, None]

    x = np.broadcast_to(x0, (M, Jsim, K_SLOW)).copy()
    out = np.empty((M, Jsim, N_REC, K_SLOW))
    out[:, :, 0] = x
    innov = sig * np.sqrt(1.0 - phi ** 2)
    eta = innov * U[None, :, 0]                    # non-stationary init

    def rhs(xx):
        return _adv(xx) - xx + F_F - (th1 + th2 * xx) + eta

    for w in range(1, N_REC):
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * DT_WIN * k1)
        k3 = rhs(x + 0.5 * DT_WIN * k2)
        k4 = rhs(x + DT_WIN * k3)
        x = x + (DT_WIN / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Runaway guard. Under the box prior this never fires -- the
        # dynamics stay within |x| < 9 -- so results are unaffected. It
        # matters only for unbounded priors, where theta2 < -1 turns the
        # -x term into positive feedback and trajectories diverge;
        # clipping at 50 leaves the L96 attractor (|x| <~ 15 at F = 10)
        # untouched while keeping the outputs finite and far from the
        # data, where Gaussian-kernel gradients vanish anyway.
        x = np.clip(x, -50.0, 50.0)
        out[:, :, w] = x
        eta = phi * eta + innov * U[None, :, w]    # updated AFTER the step

    return np.nan_to_num(out, nan=50.0, posinf=50.0, neginf=-50.0
                         ).reshape(M, Jsim, N_REC * K_SLOW)


def generate_observation_ws(rng, n_series=10):
    """One observation set: n = 10 series from the model at theta*,
    from the fixed initial state X0_FIXED. Returns (x0, Y), Y : (n, 360)."""
    ts = THETA_STAR
    th = np.array([[ts[p] for p in PARAM_NAMES]])
    U = rng.standard_normal((n_series, N_REC, K_SLOW))
    return X0_FIXED.copy(), simulate(th, X0_FIXED, U)[0]


def simulate_rk4_sens(theta, x0, U):
    """
    Same simulator as simulate(), for ONE parameter vector, returning also
    the exact forward sensitivities dx/dtheta -- needed by the
    gradient-based baseline (adSGLD).

    Everything is differentiated analytically: the map is a
    composition of RK4 stages, so the sensitivity propagates stage by
    stage. Two coupled Jacobians are carried,

        Jx = dx/dtheta      (m, K, 4)
        Je = deta/dtheta    (m, K, 4)     [sigma_e and phi only]

    with the AR(1) sensitivity
        eta_{w+1} = phi eta_w + innov U_w,   innov = sigma_e sqrt(1-phi^2)
        d eta_{w+1}/d sigma_e = phi * d eta_w/d sigma_e + sqrt(1-phi^2) U_w
        d eta_{w+1}/d phi     = eta_w + phi * d eta_w/d phi
                                - sigma_e phi / sqrt(1-phi^2) * U_w
    and the RK4-stage Jacobian of the drift
        d/dx [adv(x) - x + F - (th1 + th2 x) + eta] . V
            = J_adv(x) V - V - th2 V,
        d/d th1 = -1,   d/d th2 = -x,   d/d(sigma_e, phi) via Je.

    theta : (4,) natural space; U : (m, N_REC, K)
    returns X : (m, N_REC*K), S : (m, N_REC*K, 4)
    """
    th1, th2, sig = float(theta[0]), float(theta[1]), float(theta[2])
    phi = float(theta[3]) if len(theta) > 3 else PHI_FIXED
    m = U.shape[0]
    root = np.sqrt(max(1.0 - phi ** 2, 1e-12))
    innov = sig * root
    d_innov_dsig = root
    d_innov_dphi = -sig * phi / root

    x = np.broadcast_to(x0, (m, K_SLOW)).copy()
    Jx = np.zeros((m, K_SLOW, 4))    # internally 4-wide
    eta = innov * U[:, 0]
    Je = np.zeros((m, K_SLOW, 4))
    Je[:, :, 2] = d_innov_dsig * U[:, 0]
    Je[:, :, 3] = d_innov_dphi * U[:, 0]

    X = np.empty((m, N_REC, K_SLOW))
    S = np.empty((m, N_REC, K_SLOW, 4))
    X[:, 0] = x
    S[:, 0] = Jx                                  # IC is theta-independent

    def rhs(xx):
        return _adv(xx) - xx + F_F - (th1 + th2 * xx) + eta

    def drhs(xx, Jxx):
        JAV = (-np.roll(Jxx, 1, 1)
               * (np.roll(xx, 2, -1) - np.roll(xx, -1, -1))[..., None]
               - np.roll(xx, 1, -1)[..., None]
               * (np.roll(Jxx, 2, 1) - np.roll(Jxx, -1, 1)))
        out = JAV - Jxx - th2 * Jxx + Je
        out[:, :, 0] -= 1.0                       # d/d th1
        out[:, :, 1] -= xx                        # d/d th2
        return out

    for w in range(1, N_REC):
        k1, Jk1 = rhs(x), drhs(x, Jx)
        a2, Ja2 = x + 0.5 * DT_WIN * k1, Jx + 0.5 * DT_WIN * Jk1
        k2, Jk2 = rhs(a2), drhs(a2, Ja2)
        a3, Ja3 = x + 0.5 * DT_WIN * k2, Jx + 0.5 * DT_WIN * Jk2
        k3, Jk3 = rhs(a3), drhs(a3, Ja3)
        a4, Ja4 = x + DT_WIN * k3, Jx + DT_WIN * Jk3
        k4, Jk4 = rhs(a4), drhs(a4, Ja4)
        x_new = x + (DT_WIN / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        Jx = Jx + (DT_WIN / 6.0) * (Jk1 + 2 * Jk2 + 2 * Jk3 + Jk4)
        clipped = np.abs(x_new) > 50.0        # exact subgradient of clip
        x = np.clip(x_new, -50.0, 50.0)
        Jx[clipped] = 0.0
        X[:, w] = x
        S[:, w] = Jx
        Je_new = phi * Je
        Je_new[:, :, 3] += eta                    # d(phi eta_w)/d phi
        Je_new[:, :, 2] += d_innov_dsig * U[:, w]
        Je_new[:, :, 3] += d_innov_dphi * U[:, w]
        eta = phi * eta + innov * U[:, w]
        Je = Je_new

    # internal Jacobians are always 4-wide (incl. phi); in 3-param mode
    # the phi column is dropped so the caller sees exactly D_PAR components
    return (X.reshape(m, -1),
            S.reshape(m, N_REC * K_SLOW, 4)[:, :, :D_PAR])


# ---- summary statistics (second ABC baseline) ----------------------------
def summary_stats(X):
    """
    The 6-dimensional Hakkarainen-style summary set used by the
    summary-based SMC-ABC baseline: per series, averaged over the ring
    index k,

        1. mean_k mean_t x_k(t)
        2. mean_k Var_t x_k(t)                     (central 2nd moment)
        3. mean_k cov(x_k(t), x_k(t+1))            auto-cov, lag 1
        4. mean_k cov(x_k(t), x_{k+1}(t))          neighbour, lag 0
        5. mean_k cov(x_k(t), x_{k+1}(t+1))        neighbour, lag 1 (right)
        6. mean_k cov(x_{k+1}(t), x_k(t+1))        neighbour, lag 1 (left)

    Covariances are biased (1/T), with means taken over the
    lag-truncated windows. The statistics are NOT standardized, so the
    Euclidean distance over them is dominated by the largest-scale
    components.

    X : (..., N_REC*K) -> (..., 6)
    """
    sh = X.shape[:-1]
    x = np.swapaxes(np.asarray(X).reshape(*sh, N_REC, K_SLOW), -1, -2)

    def cov_lag(a, b, lag):
        if lag == 0:
            return (a * b).mean(-1) - a.mean(-1) * b.mean(-1)
        aa, bb = a[..., lag:], b[..., :-lag]
        return (aa * bb).mean(-1) - aa.mean(-1) * bb.mean(-1)

    xn = np.roll(x, -1, axis=-2)                      # x_{k+1}
    return np.stack([
        x.mean(-1).mean(-1),
        ((x - x.mean(-1, keepdims=True)) ** 2).mean(-1).mean(-1),
        cov_lag(x, x, 1).mean(-1),
        cov_lag(x, xn, 0).mean(-1),
        cov_lag(x, xn, 1).mean(-1),
        cov_lag(xn, x, 1).mean(-1)], axis=-1)


# ---- box handling for the samplers ---------------------------------------
def reflect_box(theta):
    span = BOX_HI - BOX_LO
    for _ in range(8):
        theta = np.where(theta < BOX_LO, 2.0 * BOX_LO - theta, theta)
        theta = np.where(theta > BOX_HI, 2.0 * BOX_HI - theta, theta)
        if np.all((theta >= BOX_LO) & (theta <= BOX_HI)):
            break
    return np.clip(theta, BOX_LO, BOX_HI)   # exact edge if still outside


def sample_box(rng, size):
    return BOX_LO + (BOX_HI - BOX_LO) * rng.uniform(size=(size, D_PAR))

