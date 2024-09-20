from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from ..core import DenseBlock, TransitionBlock


class DenseNet(nn.Module):
    """Densenet based on [d2l's implementation](https://d2l.ai/chapter_convolutional-modern/densenet.html)."""

    num_classes: int
    num_features: int = 64
    growth_rate: int = 32
    stage_sizes: tuple = (4, 4, 4, 4)
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(
        self,
        x: jax.Array,  # [B, ...spatial dims..., C]
        training: bool = False,
    ):
        d = x.ndim - 2  # num spatial dims
        bn = Partial(
            nn.BatchNorm,
            use_running_average=not training,
            axis_name="batch",
        )
        x = nn.Conv(
            self.num_features,
            kernel_size=(7,) * d,
            strides=(2,) * d,
            padding="SAME",
            name="conv_init",
        )(x)
        x = bn(name="bn_init")(x)
        x = self.act_fn(x)
        x = nn.max_pool(
            x,
            window_shape=(3,) * d,
            strides=(2,) * d,
            padding="SAME",
        )
        for i, num_blks in enumerate(self.stage_sizes):
            x = DenseBlock(num_blks, self.growth_rate)(x, training)
            num_features = self.num_features + (num_blks * self.growth_rate)
            # add transition layer to halve features between blocks
            if i + 1 != len(self.stage_sizes):  # skip last block
                num_features //= 2
                x = TransitionBlock(num_features)(x, training)
        x = bn()(x)
        x = self.act_fn(x)
        spatial_dims = jnp.arange(x.ndim)[1:-1]
        x = jnp.mean(x, axis=spatial_dims)
        return nn.Dense(self.num_classes)(x)
