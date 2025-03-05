from typing import Optional

import flax.linen as nn
from jax import Array
import jax.numpy as jnp

from dl4bi.core.transformer import TransformerEncoderBlock
from dl4bi.core.mlp import MLP


class FixedLocationTransfomer(nn.Module):
    """A transformer encoder inspired by ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).
    This transformer is used to propagate the location information for the attention bias
    and provide a head module to ensure corrent shape for the output

    Args:
        num_blks: The number of blocks to use.
        blk: An encoder block.
        norm: Final normalization module used before output.

    Returns:
        Input transformed by the encoder.
    """

    num_blks: int = 2
    embed: nn.Module = MLP([128, 64], nn.gelu)
    blk: nn.Module = TransformerEncoderBlock()
    norm: nn.Module = nn.LayerNorm()
    head: nn.Module = MLP([128, 1], nn.gelu)

    @nn.compact
    def __call__(
        self,
        x: Array,
        valid_lens: Optional[Array] = None,
        training: bool = False,
        **kwargs,
    ):
        if "s" in kwargs:
            s_batched = jnp.repeat(kwargs["s"][None, ...], repeats=x.shape[0], axis=0)
            kwargs["qs_s"], kwargs["ks_s"] = s_batched, s_batched
        x = self.embed(x)
        for _ in range(self.num_blks):
            x, _ = self.blk.copy()(x, valid_lens, training, **kwargs)
        if self.blk.pre_norm:
            x = self.norm(x)
        return self.head(x)
