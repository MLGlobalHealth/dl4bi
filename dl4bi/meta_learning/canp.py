from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..core.attention import MultiHeadAttention
from ..core.mlp import MLP
from .model_output import DiagonalMVNOutput


class CANP(nn.Module):
    """The Conditional Attentive Neural Process as detailed in [Attentive Neural Processes](https://arxiv.org/abs/1901.05761) and [Conditional Neural Processes](https://arxiv.org/pdf/1807.01613).

    This implementation is based on Google's official implementation [here]
    (https://github.com/google-deepmind/neural-processes/tree/master) and the
    hyperparameters follow Figure 8 on page 12 in [Attentive Neural Processes]
    (https://arxiv.org/abs/1901.05761) for comparison to the original Neural
    Process.

    .. note::
        The paper does not indicate that there are any projection matrices for
        queries, keys, values in MultiHeadAttention, but does specify a linear
        projection for outputs. On the other hand, the code implementation
        uses a 2-layer MLP for queries and keys, and nothing for values or
        outputs. Here, we follow the standard MultiHeadAttention setup where all
        projection matrices are single layer linear projections.

    Args:
        embed_s: An embedding module for locations.
        enc_det: An encoder for the deterministic path.
        self_attn_det: A self attention module for the deterministic path.
        cross_attn: A cross attention module used in decoding.
        dec: A decoder for test locations, aka a prediction head.
        output_fn: A function that transforms the model output into
            a form that can be consumed by loss functions.

    Returns:
        An instance of a `CANP`.
    """

    embed_s: nn.Module = MLP([128] * 2)
    enc_det: nn.Module = MLP([128] * 3)
    self_attn_det: nn.Module = MultiHeadAttention(
        proj_qs=MLP([128]),
        proj_ks=MLP([128]),
        proj_vs=MLP([128]),
        proj_out=MLP([128]),
        num_heads=8,
    )
    cross_attn: nn.Module = MultiHeadAttention(
        proj_qs=MLP([128]),
        proj_ks=MLP([128]),
        proj_vs=MLP([128]),
        proj_out=MLP([128]),
        num_heads=8,
    )
    dec: nn.Module = MLP([128] * 4 + [2])
    output_fn: Callable = DiagonalMVNOutput.from_conditional_np

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,  # [B, L_ctx, D_s]
        f_ctx: jax.Array,  # [B, L_ctx, D_f]
        s_test: jax.Array,  # [B, L_test, D_s]
        mask_ctx: Optional[jax.Array] = None,  # [B, K]
        training: bool = False,
        **kwargs,
    ):
        r_ctx = self.encode_deterministic(s_ctx, f_ctx, mask_ctx, training)
        return self.decode(r_ctx, s_ctx, s_test, mask_ctx, training)

    def encode_deterministic(
        self,
        s_ctx: jax.Array,  # [B, L_ctx, D_s]
        f_ctx: jax.Array,  # [B, L_ctx, D_f]
        mask_ctx: Optional[jax.Array] = None,  # [B, L_ctx]
        training: bool = False,
    ):
        s_f_ctx = jnp.concatenate([s_ctx, f_ctx], -1)
        s_f_ctx_embed = self.enc_det(s_f_ctx, training)
        r_ctx, _ = self.self_attn_det(
            s_f_ctx_embed,
            s_f_ctx_embed,
            s_f_ctx_embed,
            mask_ctx,
            training,
        )
        return r_ctx

    def decode(
        self,
        r_ctx: jax.Array,  # [B, d_ffn]
        s_ctx: jax.Array,  # [B, L_ctx, D_s]
        s_test: jax.Array,  # [B, L_test, D_s]
        mask_ctx: Optional[jax.Array],  # [B, L_ctx]
        d_f: int,
        training: bool = False,
    ):
        r, _ = self.cross_attn(
            self.embed_s(s_test),
            self.embed_s(s_ctx),
            r_ctx,
            mask_ctx,
            training,
        )  # [B, L_test, d_ffn]
        q = jnp.concatenate([r, s_test], -1)  # [B, L_test, d_ffn + D_s]
        output = self.dec(q, training)
        return self.output_fn(output)
