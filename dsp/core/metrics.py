from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax
from jax._src.numpy.util import promote_dtypes_inexact
from jax.scipy.stats import norm
from jax.typing import ArrayLike


@jit
def l2_dist_sq(x: jax.Array, y: jax.Array) -> jax.Array:
    """L2 distance between two [..., D] arrays.

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        Matrix of all pairwise distances.
    """
    x, y = prepare_dims(x, y)
    return (x**2).sum(-1)[:, None] + (y**2).sum(-1).T - 2 * x @ y.T


@jit
def prepare_dims(x: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Prepares dims for use in kernel functions.

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        Two `[N, D]` dimensional arrays.
    """
    if x.ndim == 1:
        x = x[:, jnp.newaxis]
    if y.ndim == 1:
        y = y[:, jnp.newaxis]
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1, y.shape[-1])
    return x, y


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


@partial(jit, static_argnames=["num_bins"])
def mean_absolute_calibration_error(
    f_true: jax.Array,
    f_mu: jax.Array,
    f_std: jax.Array,
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
    upper = norm.ppf(0.5 + p / 2.0)
    z = ((f_mu - f_true) / f_std)[..., None]
    covered = jnp.logical_and(z >= lower, z <= upper)
    p_covered = jnp.mean(covered, axis=range(1, covered.ndim - 2))
    return jnp.mean(jnp.abs(p_covered - p), -1)
