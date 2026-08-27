"""
metrics.py
==========
Quantitative distances between a sample-based posterior approximation and
the exact grid reference. All metrics are chosen to be exact/deterministic
in 1-D, so no classifier training (C2ST) or KDE smoothing noise enters the
bias measurement.

Headline metric
---------------
Wasserstein-1 via the quantile representation, exact in 1-D:
    W1(P_hat, Pi_ref) = int_0^1 | F_hat^{-1}(p) - F_ref^{-1}(p) | dp
approximated by evaluating both quantile functions at the K midpoint levels
p_k = (k - 1/2)/K, where K = number of samples -- i.e. the empirical quantile
function is simply the sorted sample, and the reference quantile function is
interpolated from the quadrature CDF (reference.ReferencePosterior.quantile).

Secondary metrics
-----------------
Relative errors of posterior mean and std separately. The std error is
tracked on purpose: a systematic under/over-dispersion by the linearization
is an uncertainty-quantification finding that feeds the coverage study.
"""

import numpy as np


def w1_to_reference(samples, ref):
    """Exact 1-D W1 between the empirical measure of `samples` and `ref`."""
    s = np.sort(np.asarray(samples))
    K = len(s)
    p = (np.arange(K) + 0.5) / K
    return float(np.mean(np.abs(s - ref.quantile(p))))


def moment_errors(samples, ref):
    """Relative errors of mean and std against the quadrature reference."""
    m, sd = float(np.mean(samples)), float(np.std(samples))
    scale = max(abs(ref.mean), ref.std)   # avoid division by ~0 posterior mean
    return {
        "mean_err": abs(m - ref.mean) / scale,
        "std_err": abs(sd - ref.std) / ref.std,
        "sample_mean": m,
        "sample_std": sd,
        "ref_mean": ref.mean,
        "ref_std": ref.std,
    }
