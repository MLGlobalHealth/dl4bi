from collections import defaultdict
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scoringrules as sr
from jax import jit, lax
from jax._src.numpy.util import promote_dtypes_inexact
from jax.scipy.stats import norm
from jax.typing import ArrayLike


def mvn_logpdf(
    x: ArrayLike,
    mean: ArrayLike,
    cov: ArrayLike,
    is_tril: bool = False,
) -> ArrayLike:
    """MVN logpdf supporting tril based on JAX implementation [here](https://github.com/google/jax/blob/main/jax/_src/scipy/stats/multivariate_normal.py#L25-L73).

    Args:
      x: arraylike, value at which to evaluate the PDF
      mean: arraylike, centroid of distribution
      cov: arraylike, covariance matrix of distribution

    Returns:
      array of logpdf values.
    """
    x, mean, cov = promote_dtypes_inexact(x, mean, cov)
    if not mean.shape:
        return -1 / 2 * jnp.square(x - mean) / cov - 1 / 2 * (
            jnp.log(2 * np.pi) + jnp.log(cov)
        )
    else:
        n = mean.shape[-1]
        if not np.shape(cov):
            y = x - mean
            return -1 / 2 * jnp.einsum("...i,...i->...", y, y) / cov - n / 2 * (
                jnp.log(2 * np.pi) + jnp.log(cov)
            )
        else:
            if cov.ndim < 2 or cov.shape[-2:] != (n, n):
                raise ValueError("multivariate_normal.logpdf got incompatible shapes")
            L = cov if is_tril else lax.linalg.cholesky(cov)  # modified line
            y = jnp.vectorize(
                partial(lax.linalg.triangular_solve, lower=True, transpose_a=True),
                signature="(n,n),(n)->(n)",
            )(L, x - mean)
            return (
                -1 / 2 * jnp.einsum("...i,...i->...", y, y)
                - n / 2 * jnp.log(2 * np.pi)
                - jnp.log(L.diagonal(axis1=-1, axis2=-2)).sum(-1)
            )


def mean_kl_div_diag_mvn(f_mu_p, f_std_p, f_mu_q, f_std_q):
    f_var_p, f_var_q = f_std_p**2, f_std_q**2
    return 0.5 * jnp.mean(
        f_var_p / f_var_q
        + (f_mu_p - f_mu_q) ** 2 / f_var_q
        - 1
        + jnp.log(f_var_q / f_var_p)
    )


@partial(jit, static_argnames=["num_bins"])
def mean_absolute_calibration_error(
    f_true: jax.Array,  # [B, ..., D]
    f_mu: jax.Array,  # [B, ..., D]
    f_std: jax.Array,  # [B, ..., D]
    num_bins: int = 100,
):
    """
    Calculates Mean Absolute Calibration Error (MACE), which is equivalent to
    Expected Calibration Error (ECE).

    .. note::
        All arrays must be of shape `[B, ..., D]` where the last dimension
        is assumed to be a vector of independent Gaussian values.

    Args:
        f_true: Array of true function values.
        f_mu: Mean of predicted function values.
        f_std: Standard devation of predicted function values.
        num_bins: Number of bins for discretizing probability space.

    Returns:
        The MACE (ECE) of shape `[B, D]`.
    """
    p = jnp.linspace(0, 1, num_bins)
    lower = norm.ppf(0.5 - p / 2.0)
    upper = -lower
    z = ((f_mu - f_true) / f_std)[..., None]
    covered = jnp.logical_and(z >= lower, z <= upper)
    p_covered = jnp.mean(covered, axis=range(1, covered.ndim - 2))
    return jnp.mean(jnp.abs(p - p_covered), axis=-1)


def compute_inference_metrics(
    data: dict,
    predictions: dict,
    hdi_prob: float = 0.95,
    **kwargs,
):
    alpha = 1 - hdi_prob
    z_score = jnp.abs(norm.ppf(alpha / 2))
    Nc = data["valid_lens_ctx"]
    f = data["f"][Nc:]
    m = defaultdict(dict)
    for method, d in predictions.items():
        f_mu = d["f_mu"][Nc:]
        if "f_std" in d:
            f_std = d["f_std"][Nc:]
            f_lower = f_mu - z_score * f_std
            f_upper = f_mu + z_score * f_std
        else:  # tabpfn
            f_lower = d["f_lower"][Nc:]
            f_upper = d["f_upper"][Nc:]
        m["Log Likelihood (LL)"][method] = np.sum(norm.logpdf(f, f_mu, f_std))
        m["Interval Score (IS)"][method] = np.mean(
            sr.interval_score(f, f_lower, f_upper, alpha)
        )
        m["Continuous Ranked Probability Score (CRPS)"][method] = np.mean(
            sr.crps_normal(f, f_mu, f_std)
        )
        m["Coverage (CVG)"][method] = ((f >= f_lower) & (f <= f_upper)).mean()
        m["Mean Absolute Error (RMSE)"][method] = np.abs(f - f_mu).sum()
        m["Root Mean Squared Error (RMSE)"][method] = np.sqrt(
            np.square(f - f_mu).mean()
        )
    return dict(m)
