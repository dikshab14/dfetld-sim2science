"""
common/mmd.py
=============
Empirical MMD^2 machinery for MULTIVARIATE data (used by exp1_l96 and
exp2_contamination; the 1-D appendix linearization study keeps its own
closed-form module).

Conventions match the paper:
  * Gaussian kernel k(x,y) = exp(-||x-y||^2 / (2 gamma^2)), x,y in R^n.
  * Unbiased U-statistic estimator MMD^2_hat, eq. (2) of the paper:
    model-model term over j != l, data-data term over n != n', cross term
    over all (j,n). With a SINGLE observed trajectory (N = 1, the Lorenz96
    setting, as in the paper and in Pacchiardi et al. 2024), the data-data
    term is dropped: it is theta-independent, so it shifts the generalized
    posterior by a constant and cancels after normalization.
  * Bandwidth: median heuristic. With N = 1 observation it CANNOT be
    computed from the data; the standard fallback (used here) is the median
    pairwise distance among PRIOR-PREDICTIVE simulations, computed once and
    frozen for the entire experiment so bandwidth adaptation never
    confounds method comparisons.
"""

import numpy as np


def sq_dists(A, B):
    """Pairwise squared Euclidean distances. A: (a,n), B: (b,n) -> (a,b)."""
    return (np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :]
            - 2.0 * A @ B.T)


def median_heuristic_from_sims(X):
    """
    gamma = median pairwise distance among simulated samples X (S, n).
    Used when the observed data are a single vector (N = 1): see module
    docstring. Frozen after one computation.
    """
    d2 = sq_dists(X, X)
    iu = np.triu_indices(len(X), k=1)
    return float(np.sqrt(np.median(d2[iu])))
