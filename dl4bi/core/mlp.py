from collections.abc import Callable
from typing import Optional

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn
from jax.nn import initializers as init


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
        x = rearrange(x, "B H W C -> B (H W) C")
        for _ in range(self.num_blks):
            x = MLPMixerBlock(self.token_dims, self.channel_dims)(x)
        x = nn.LayerNorm()(x)
        x = jnp.mean(x, axis=1)
        return nn.Dense(self.num_cls, kernel_init=nn.initializers.zeros)(x)


class SpatialGatingUnit(nn.Module):
    """Spatial Gating Unit from [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050).

    Based on implementation [here](https://github.com/lucidrains/g-mlp-pytorch/tree/main).

    Args:
        num_heads: Number of heads where each gets its own bias term.
        gate_fn: Activation function for gate inputs.
        norm: Module used to normalize gate values.

    Returns:
        An instance of `SpatialGatingUnit`.
    """

    num_heads: int = 1
    gate_fn: Callable = lambda x: x
    norm: nn.Module = nn.LayerNorm()

    @nn.compact
    def __call__(self, x, gate_res: Optional[jax.Array] = None):
        B, L, D = x.shape
        H = 1 if gate_res is None else gate_res.shape[1]  # B L D
        bias = self.param("bias", init.constant(1.0), (1, H, L, 1))
        weights = self.param("weights", init.uniform(-1e-3 / L, 1e-3 / L), (H, L, L))
        z_1, z_2 = jnp.split(x, 2, axis=-1)
        z_2 = self.norm(z_2)
        z_2 = rearrange(z_2, "B L (H D) -> B H L D", H=H)
        z_2 = jnp.einsum("B H L D, H M L -> B H M D", z_2, weights)  # M = L
        z_2 = rearrange(z_2 + bias, "B H L D -> B L (H D)")
        if gate_res is not None:
            z_2 += gate_res
        return z_1 * self.gate_fn(z_2)


# TODO(danj): implement GatedMLPBlock for vision with patch sizes
# TODO(danj): implement causal masking variant for autoregressive tasks
# TODO(danj): try implementing Toeplitz matrices with circulant matrices --
# useful on MLM objective, but perhaps not generally?
class GatedMLPBlock(nn.Module):
    """Gated MLP Block from [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050).

    Based on implementation [here](https://github.com/lucidrains/g-mlp-pytorch/tree/main).

    Args:
        proj_in: Module that projects last dimension of input.
        proj_out: Module that projects last dimension of output.
        attn: An optional attention module used in aMLP variant.

    Returns:
        An instance of `GatedMLPBlock`.
    """

    proj_in: nn.Module = MLP([128, 64], nn.gelu)
    proj_out: nn.Module = MLP([128, 1], nn.gelu)
    attn: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, x):
        pass


class GatedMLP(nn.Module):
    """Gated MLP from [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050).

    Based on implementation [here](https://github.com/lucidrains/g-mlp-pytorch/tree/main).

    Args:
        num_blks: Number of `GatedMLPBlocks` to use.
        proj_in: Module that projects last dimension of input for blocks.
        proj_out: Module that projects last dimension of output for blocks.
        attn: An optional attention module used in aMLP variant for blocks.

    Returns:
        An instance of `GatedMLP`.
    """

    num_blks: int = 6
    proj_in: nn.Module = MLP([128, 64], nn.gelu)
    proj_out: nn.Module = MLP([128, 1], nn.gelu)
    attn: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, x):
        pass
