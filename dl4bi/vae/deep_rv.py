from collections.abc import Callable

import flax.linen as nn
from jax import Array

from dl4bi.core.model_output import VAEOutput


class DeepRV(nn.Module):
    r"""`DeepRV` learns to emulate samples from a fixed size stochastic process.

    The model learns the function $f_\mathbf{c}:(\mathbf{z})\to\mathbf{T}_{\mathbf{c}}\mathbf{z}$,
    from the latent space to the realizations of the process given conditional hyperparameters $\mathbf{c}$,
    where $\mathbf{L}_{\mathbf{c}}$ is conditioned on the hyperparameters $\mathbf{c}$.

    E.g for a Gaussian Process (GP) with a kernel $\mathcal{K}_\mathbf{c}$, and a fixed spatial structure
    $\mathbf{x}$, the model emulates the GP by learning the following function:
    $\text{Cholesky}(\mathbf{K}_\mathbf{c})\mathbf{z} \approx  \text{DeepRV}(\mathbf{z}, \mathbf{c})$,
    which subsequentially gives us:

    $\mathbf{f}_\mathbf{c}\sim\mathcal{GP}_\mathbf{c}(\cdot,\cdot) \approx \mathbf{\hat{f}}_{\mathbf{c}} =
    \text{DeepRV}(\mathbf{z}, \mathbf{c}), \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$

    The decoder submodule can be any neural network whose output
    size is `num_locations`.

    Args:
        z: latent vector samples. The first dimension is assumed to be the
            batch dimension.
        conditionals: conditional hyperparameters of the process.

    Returns:
        An instance of the `DeepRV` network.
    """

    decoder: nn.Module
    cond_stack_fn: Callable

    @nn.compact
    def __call__(self, z: Array, conditionals: Array, **kwargs):
        r"""Run module forward.

        Args:
            z: latent vector samples.
            conditionals: conditional hyperparameters of the process.

        Returns:
            $\hat{\mathbf{f}}$, an approximation of the stochastic process's realizations.
        """
        return VAEOutput(self.decode(z, conditionals, **kwargs))

    def decode(self, z: Array, conditionals: Array, **kwargs):
        return self.decoder(self.cond_stack_fn(z, conditionals), **kwargs)
