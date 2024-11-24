from dataclasses import field
from typing import List, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from sps.utils import build_grid

from ..core import MLP, ConvCNPNet, ConvDeepSet


class ConvCNP(nn.Module):
    """[The Convolutional Conditional Neural Process](https://arxiv.org/abs/1910.13556).

    Based on Yann Dubois' implementation [here](https://github.com/YannDubs/Neural-Process-Family/blob/master/npf/neuralproc/convnp.py#L26).

    Args:
        s_lower: Lower coordinate bound of grid.
        s_upper: Upper coordinate bound of grid.
        points_per_unit: Number of points per unit interval on input, which is
            used to discretize the function.
        enc: An encoder module that uses context points to infer function values
            on a grid.
        conv_net: A convolutional network used for the function representation
            on a grid.
        dec_f_mu: A module that decodes `f_mu` from a grid at test locations.
        dec_f_std: A module that decodes `f_std` from a grid at test locations.

    Returns:
        An instance of `CNP`.
    """

    s_lower: List[float] = field(default_factory=lambda: [-2.5])
    s_upper: List[float] = field(default_factory=lambda: [2.5])
    points_per_unit: int = 128
    min_std: float = 0.0
    enc: nn.Module = ConvDeepSet()
    conv_net: nn.Module = ConvCNPNet()
    dec: nn.Module = ConvDeepSet()
    head: nn.Module = MLP([128] * 3 + [2])

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,  # [B, L_ctx, D_s]
        f_ctx: jax.Array,  # [B, L_ctx, D_f]
        s_test: jax.Array,  # [B, L_test, D_s]
        valid_lens_ctx: Optional[jax.Array] = None,  # [B]
        valid_lens_test: Optional[jax.Array] = None,  # [B]
        training: bool = False,
        **kwargs,
    ):
        B = s_ctx.shape[0]
        s_dim = len(self.s_lower)
        s_grid = build_grid(
            [
                dict(start=lo, stop=up, num=int(self.points_per_unit * (up - lo)))
                for (lo, up) in zip(self.s_lower, self.s_upper)
            ]
        )  # [*P... , s_dim]
        s_grid = jnp.repeat(s_grid[None, :], B, axis=0)  # [B, *P... , s_dim]
        conv_dims = s_grid.shape[:-1]  # [B, *P...]
        s_vec = s_grid.reshape(B, -1, s_dim)  # [B, L_grid, s_dim]
        h = self.enc(s_ctx, f_ctx, s_vec, valid_lens_ctx)  # [B, L_grid, D]
        h = self.conv_net(h.reshape(conv_dims + (-1,)))  # [B, *P..., D]
        h = self.dec(s_vec, h.reshape(B, s_vec.shape[1], -1), s_test)  # [B, L_grid, D]
        f_dist = self.head(h)
        f_mu, f_std = jnp.split(f_dist, 2, axis=-1)
        f_std = self.min_std + (1 - self.min_std) * nn.softplus(f_std)
        return f_mu, f_std
