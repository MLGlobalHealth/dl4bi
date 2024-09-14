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
    ffn_dims: list[int]
    mix_dims: list[int]

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = jnp.swapaxes(x, 1, 2)
        y = MLP(self.ffn_dims, nn.gelu, name="token_mixing")(x)
        y = jnp.swapaxes(x, 1, 2)
        x = x + y
        y = nn.LayerNorm()(x)
        return x + MLP(self.mix_dims, nn.gelu, name="channel_mixing")(y)


class MLPMixer(nn.Module):
    num_cls: int
    num_blks: int
    blk: nn.Module = MLPMixerBlock([128, 128], [128, 128])
    conv: nn.Module = nn.Conv(128, (16, 16), strides=(16, 16))

    @nn.compact
    def __call__(self, x):
        x = self.conv(x)
        x = einops.rearrange(x, "B H W C -> B (H W) C")
        for _ in range(self.num_blks):
            x = self.blk.copy()(x)
        x = nn.LayerNorm()(x)
        x = jnp.mean(x, axis=1)
        return nn.Dense(self.num_cls, kernel_init=nn.initializers.zeros)(x)

    @classmethod
    def build(
        cls,
        patch_size: int,
        conv_dim: int,
        ffn_dim: int,
        mix_dim: int,
        num_blks: int,
        num_cls: int,
    ):
        s = patch_size
        conv = nn.Conv(conv_dim, (s, s), strides=(s, s))
        blk = MLPMixerBlock([ffn_dim] * 2, [mix_dim] * 2)
        return cls(num_cls, num_blks, blk, conv)
