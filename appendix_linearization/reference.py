"""
reference.py
============
Rung 1 of the comparison ladder: the EXACT MMD-Bayes posterior on a fine grid.

Because model.mmd2_exact is available in closed form (eqs. (M4)-(M6)), the
generalized posterior

    pi(theta | Y)  prop.  exp( log_prior(theta) - beta * MMD^2(P_theta, P_hat) )

can be evaluated pointwise with zero Monte Carlo error, normalized by
trapezoidal quadrature, and summarized (mean, std, CDF, quantiles) to
machine precision on a 20k-point grid. The gold reference therefore carries
no sampling error.

The CDF/quantile machinery is what metrics.w1_to_reference uses to compute
1-D Wasserstein-1 distances exactly via the quantile representation
    W1(P, Q) = int_0^1 |F_P^{-1}(p) - F_Q^{-1}(p)| dp.
"""

import numpy as np

from model import mmd2_exact, log_prior


class ReferencePosterior:
    """Exact grid posterior with quadrature-based summaries."""

    def __init__(self, y, alpha, sigma, gamma, beta, m0, s0,
                 grid_lo, grid_hi, grid_size):
        self.grid = np.linspace(grid_lo, grid_hi, grid_size)
        dx = self.grid[1] - self.grid[0]

        # un-normalized log posterior on the grid
        log_post = (log_prior(self.grid, m0, s0)
                    - beta * mmd2_exact(self.grid, y, alpha, sigma, gamma))

        # normalize stably (subtract max before exponentiating)
        log_post -= log_post.max()
        w = np.exp(log_post)
        Z = np.trapezoid(w, self.grid)
        self.pdf = w / Z

        # sanity check that the grid actually contains the posterior mass:
        # density at both ends must be negligible, else grid_lo/hi too tight.
        edge = max(self.pdf[0], self.pdf[-1])
        if edge > 1e-8 * self.pdf.max():
            raise ValueError(
                "Posterior mass touches the grid boundary "
                f"(edge density ratio {edge / self.pdf.max():.2e}); "
                "widen grid_lo/grid_hi.")

        # moments by quadrature
        self.mean = float(np.trapezoid(self.grid * self.pdf, self.grid))
        self.var = float(np.trapezoid(
            (self.grid - self.mean) ** 2 * self.pdf, self.grid))
        self.std = float(np.sqrt(self.var))

        # CDF (cumulative trapezoid), clipped and rescaled to [0, 1] exactly
        cdf = np.concatenate(
            [[0.0], np.cumsum(0.5 * (self.pdf[1:] + self.pdf[:-1]) * dx)])
        self.cdf = cdf / cdf[-1]

    def quantile(self, p):
        """
        Inverse CDF by linear interpolation on the grid.
        p : array of probabilities in (0, 1)  ->  array of theta values.
        np.interp requires the x-coordinates (here the CDF) to be increasing;
        the CDF of a continuous density on a fine grid is, up to flat regions
        in the far tails, which interp handles by returning the left edge --
        harmless because no probability mass lives there.
        """
        return np.interp(p, self.cdf, self.grid)
