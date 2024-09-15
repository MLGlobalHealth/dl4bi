from collections.abc import Callable

import einops
import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    dims: list[int]
    act_fn: Callable = nn.relu
    p_dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, training: bool = False):
        for dim in self.dims[:-1]:
            x = nn.Dense(dim, dtype=self.dtype)(x)
            x = self.act_fn(x)
            x = nn.Dropout(self.p_dropout, deterministic=not training)(x)
        return nn.Dense(self.dims[-1], dtype=self.dtype)(x)


class MLPMixerBlock(nn.Module):
    token_dims: list[int]
    channel_dims: list[int]

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = jnp.swapaxes(x, 1, 2)
        y = MLP(self.token_dims, nn.gelu, name="token_mixing")(x)
        y = jnp.swapaxes(x, 1, 2)
        x = x + y
        y = nn.LayerNorm()(x)
        return x + MLP(self.channel_dims, nn.gelu, name="channel_mixing")(y)


class MLPMixer(nn.Module):
    num_cls: int
    num_blks: int
    token_dims: list[int]
    channel_dims: list[int]
    patch_size: int = 1
    conv_dim: int = 128

    @nn.compact
    def __call__(self, x):
        s = self.patch_size
        x = nn.Conv(self.conv_dim, (s, s), strides=(s, s))(x)
        x = einops.rearrange(x, "B H W C -> B (H W) C")
        for _ in range(self.num_blks):
            x = MLPMixerBlock(self.token_dims, self.channel_dims)(x)
        x = nn.LayerNorm()(x)
        x = jnp.mean(x, axis=1)
        return nn.Dense(self.num_cls, kernel_init=nn.initializers.zeros)(x)
