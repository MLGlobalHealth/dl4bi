"""
RBF binomial model. This is nearly identical to the survey model.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from sps.kernels import rbf


def kernel(x, y, /, *, var, ls, **_):
    return rbf(x, y, var, ls)


jitter = 1e-4  # note this is in fact N(0, s2=jitter) independent noise


def spatial_effect(s: jax.Array, *, sample_shape: tuple[int, ...] = ()):
    """
    Definition of the spatial effect underlying the observation model.

    If sample_shape is not (), the kernel parameters are shared across samples.
    """
    ls = numpyro.sample("ls", dist.InverseGamma(3, 3))
    var = numpyro.sample("var", dist.InverseGamma(3, 3))

    cov = kernel(s, s, var=var, ls=ls)
    L = cov.shape[0]
    cov = cov + jitter * jnp.eye(L)

    y = numpyro.sample(
        "y",
        dist.MultivariateNormal(0, cov),
        sample_shape=sample_shape,
    )

    return y


def prevalence(y, x=None):
    *sample_shape, L = y.shape
    sample_shape = tuple(sample_shape)

    b0 = numpyro.sample("b0", dist.Normal(-2, 5), sample_shape=sample_shape)
    b0 = b0[..., None]  # make broadcastable to [sample_shape, L]

    if x is not None:
        D = x.shape[-1]
        assert x.shape == (L, D)

        b = numpyro.sample("b", dist.Normal(0, 5), sample_shape=(*sample_shape, D))
        bx = jnp.einsum("...d, ...ld-> ...l", b, x)
        assert bx.shape == (*sample_shape, L)

        z = numpyro.deterministic("z", y + b0 + bx)
    else:
        z = numpyro.deterministic("z", y + b0)

    numpyro.deterministic("theta", jax.nn.sigmoid(z))
    return z


def model(
    s: jax.Array,  # [L, Ds]
    n: jax.Array,  # [L]
    n_pos: jax.Array | None,  # [L]
    x: jax.Array | None = None,  # [L, Dx]
    *,
    sample_shape: tuple[int, ...] = (),
):
    """
    MBG survey model.

    Setting `sample_shape` to !=() will produce samples with the same kernel
    parameters in the spatial effect, so that they are not iid but only iid
    given the kernel parameters.
    """
    y = spatial_effect(s, sample_shape=sample_shape)
    z = prevalence(y, x)

    numpyro.sample(
        "n_pos",
        dist.BinomialLogits(total_count=n, logits=z),
        obs=n_pos,
    )
