"""Vision Transformer building blocks."""

from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from .model_output import MultinomialOutput
from .transformer import TransformerEncoderBlock


# TODO(danj): add registers!
# TODO(danj): mnist benchmark
# TODO(danj): try SSL with DINOv2
class ViT(nn.Module):
    """A compact Vision Transformer classifier."""

    num_blks: int = 6
    patch: nn.Module = nn.Conv(
        features=128,
        kernel_size=(4, 4),
        strides=(4, 4),
        padding="VALID",
    )
    blk: nn.Module = TransformerEncoderBlock()
    p_dropout: float = 0.0
    output_fn: Callable = MultinomialOutput.from_activations

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False, **kwargs):
        """Embed image patches, apply transformer blocks, and classify."""
        x = self.patch(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        cls_token = self.param(
            "cls_token",
            nn.initializers.constant(0),
            (1, 1, C),
        )
        cls_tokens = jnp.tile(cls_token, (B, 1, 1))
        x = jnp.concat([cls_tokens, x], axis=1)
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.lecun_normal(),
            (1, H * W, C),
        )
        x += pos_embed
        x = nn.Dropout(self.p_dropout, deterministic=not training)(x)
        for _ in range(self.num_blks):
            x = self.blk.copy()(x, training=training)
        cls_tokens = x[:, 0]
        output = self.head(cls_tokens)
        return self.output_fn(output)
