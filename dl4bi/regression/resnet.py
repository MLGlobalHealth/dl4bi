from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from ..core import ResNetBlock


class ResNet(nn.Module):
    """ResNet v1.5 based on Flax [example](https://github.com/google/flax/blob/main/examples/imagenet/models.py)."""

    num_classes: int
    num_features: int = 64
    stage_sizes: tuple = (2, 2, 2, 2)  # ResNet18
    act_fn: Callable = nn.relu
    blk_cls: Any = ResNetBlock

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
        conv = Partial(nn.Conv, use_bias=False)
        x = conv(
            self.num_features,
            kernel_size=(7,) * d,
            strides=(2,) * d,
            padding=[(3,) * d] * d,
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
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                # after the first stage, the first block in each stage halves
                # the spatial dimensions and doubles the number of features
                strides = (2,) * d if i > 0 and j == 0 else (1,) * d
                x = self.blk_cls(self.num_features * 2**i, strides)(x)
        spatial_dims = jnp.arange(x.ndim)[1:-1]
        x = jnp.mean(x, axis=spatial_dims)
        return nn.Dense(self.num_classes)(x)
