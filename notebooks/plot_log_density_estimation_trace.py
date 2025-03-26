from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

index = 13
file = "tmp/2025-03-10 19:31:40.462653.npy"

file = Path(file)

log_densities = jnp.load(file)


def estimate(n):
    return jax.nn.logsumexp(log_densities[:n], axis=0) - jnp.log(
        jnp.astype(n, jnp.float32)
    )


ns = jnp.arange(1, log_densities.shape[0])
estimates = jnp.array([estimate(n) for n in ns]).squeeze()


plt.plot(ns, estimates[:, index])
plt.title("Running estimates for the random-order induced log density")
plt.show()


plt.plot(log_densities[:, index].squeeze())
plt.title("Log-density samples")
plt.show()
