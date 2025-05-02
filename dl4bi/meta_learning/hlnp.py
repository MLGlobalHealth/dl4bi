from collections.abc import Callable

import flax.linen as nn

from ..core.mlp import MLP
from ..core.model_output import DiagonalMVNOutput
from .steps import likelihood_train_step, likelihood_valid_step


class HLNP(nn.Module):
    """Hierarchical Latent Neural Process (HLNP).

    Args:
        norm: A module used for normalization between blocks.
        head: Transforms the tokens into model output.
        output_fn: A function that transforms the model output into
            a form that can be consumed by loss functions.
        train_step: What training step to use.
        valid_step: What validation step to use.

    Returns:
        An instance of the `HLNP` model.
    """

    norm: nn.Module = nn.LayerNorm()
    head: nn.Module = MLP([256, 64, 2], nn.gelu)
    output_fn: Callable = DiagonalMVNOutput.from_activations
    train_step: Callable = likelihood_train_step
    valid_step: Callable = likelihood_valid_step

    @nn.compact
    def __call__(self):
        pass
