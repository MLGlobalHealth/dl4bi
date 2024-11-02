from collections.abc import Callable
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import vmap

from .utils import outer_subtract


class DistanceBias(nn.Module):
    """A Distance Bias module.

    Args:
        func:

    Returns:
        An instance of `DistanceBias` module.
    """

    transform: Callable = lambda d: jnp.linalg.norm(d, axis=-1)  # L1
    num_heads: int = 1

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,  # [B, L_ctx, D_S]
        f_ctx: jax.Array,  # [B, L_ctx, D_F]
        s_test: jax.Array,  # [B, L_test, D_S]
        valid_lens_ctx: Optional[jax.Array] = None,  # [B]
        valid_lens_test: Optional[jax.Array] = None,  # [B]
    ):
        init = nn.initializers.constant(1)
        bias_scale = self.param("bias_scale", init, (1, self.num_heads, 1, 1))
        # negative because you want closer locations to have higher values
        bias = -self.transform(vmap(outer_subtract)(s_ctx, s_test))
        bias = jnp.repeat(bias[:, None, ...], self.num_heads, axis=1)
        return bias_scale * bias
