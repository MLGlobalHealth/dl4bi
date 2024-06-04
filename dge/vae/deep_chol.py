import flax.linen as nn
import jax.numpy as jnp
from jax import Array


class DeepChol(nn.Module):
    r"""`DeepChol` learns to approximate the function $f_\theta:(\mathbf{z},\text{var},\text{ls})\to\mathbf{Lz}$.

    Note that $\mathbf{L}$, and thus $\mathbf{Lz}$, is conditional on the kernel
    and the locations where it is sampled. If we label the kernel $\mathcal{K}$,
    and the locations where it is sampled $\mathbf{x}$, we can write the function
    more accurately as:

    $f_{\theta\mid\mathcal{K},\mathbf{x}}(\mathbf{z},\text{var},\text{ls})\approx\langle\text{Cholesky}(\mathcal{K}(\mathbf{x},\mathbf{x}\mid\text{var},\text{ls})),\mathbf{z}\rangle$

    The decoder submodule can be any neural network whose output
    size is `num_locations`.

    Args:
        z: Standard normal samples with one `z` for each location. The first
            dimension is assumed to be the batch dimension.
        var: Variance to use when calculating `f`.
        ls: lengthscale to use when calculating `f`.

    Returns:
        An instance of the `DeepChol` network.
    """

    decoder: nn.Module

    @nn.compact
    def __call__(self, z: Array, var: float, ls: float):
        r"""Run module forward.

        Args:
            rng: A psuedo-random number generator.
            var: The variance for the GP.
            ls: The lengthscale for the GP.

        Returns:
            $\hat{\mathbf{f}}$, an approximation of $\mathbf{Lz}$.
        """
        batch_size = z.shape[0]
        var = jnp.full((batch_size, 1), var)
        ls = jnp.full((batch_size, 1), ls)
        x = jnp.hstack([z, var, ls])
        return self.decoder(x)
