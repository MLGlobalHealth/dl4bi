import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from sps.kernels import rbf

var_dist = dist.Delta(1)
ls_dist = dist.Beta(3, 7)
kernel = rbf
jitter = 1e-5  # note this is in fact N(0, s2=jitter) noise


def model(
    s: jax.Array,
    Np: jax.Array,
    Nt: jax.Array,
):
    B, L, D = s.shape

    # isn't it better to account for scale later on in the NP framework? or both?
    var = numpyro.sample("var", var_dist)
    ls = numpyro.sample("ls", ls_dist)

    K = kernel(s, s, var, ls) + jitter * jnp.eye(L)

    y = numpyro.sample(
        "f_s",
        dist.MultivariateNormal(0, K),
    )

    s = numpyro.sample("s", dist.HalfNormal(50))
    b = numpyro.sample("b", dist.Normal(0, 1))

    logit_theta = b + s * y
    numpyro.deterministic("theta", jax.nn.sigmoid(logit_theta))

    numpyro.sample(
        "Np", dist.BinomialLogits(total_count=Nt, logits=logit_theta), obs=Np
    )
