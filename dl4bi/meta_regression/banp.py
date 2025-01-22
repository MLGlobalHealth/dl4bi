from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import jit, random
from jax.lax import stop_gradient as no_grad

from ..core import MLP, MultiHeadAttention, bootstrap, mask_from_valid_lens
from .transform import diagonal_mvn


class BANP(nn.Module):
    """The Bootstrapping Attentive Neural Process as detailed in [Bootstrapping Neural Processes](https://arxiv.org/abs/2008.02956).

    This implementation is based on the official implementation
    [here](https://github.com/juho-lee/bnp/tree/master), although
    we use the hyperparameters specified in Figure 8 on page 12 of
    [Attentive Neural Processes](https://arxiv.org/abs/1901.05761)
    to keep comparisons among models consistent.

    .. note::
        The Attentive Neural Processes paper does not indicate that there are
        any projection matrices for queries, keys, values in MultiHeadAttention,
        but does specify a linear projection for outputs. On the other hand, the
        code implementation uses a 2-layer MLP for queries and keys, and nothing
        for values or outputs. Here, we follow the standard MultiHeadAttention
        setup where all projection matrices are single layer linear projections.

    Args:
        num_samples: The number of samples to use for bootstrapping.
        embed_s: An embedding module for locations.
        enc_det: An encoder for the deterministic path.
        self_attn_det: A self attention module for the deterministic path.
        cross_attn: A cross attention module used in decoding.
        dec_hid: The first stage of decoding at test points.
        dec_boot: A decoding module that integrates bootstrapped samples.
        dec_dist: Decodes the hidden state into model output.
        output_fn: A function that transforms the model output into
            a form that can be consumed by loss functions.

    Returns:
        An instance of a `BANP`.
    """

    num_samples: int = 4
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
    dec_hid: nn.Module = MLP([128])
    dec_boot: nn.Module = MLP([128] * 2)
    dec_dist: nn.Module = MLP([128] * 3 + [2])
    output_fn: Callable = diagonal_mvn

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
        rep = jit(lambda x: jnp.repeat(x, self.num_samples, axis=0))
        r_ctx = self.encode_deterministic(s_ctx, f_ctx, valid_lens_ctx, training)
        s_ctx_boot, f_ctx_boot, valid_lens_ctx_boot = self.sample_with_replacement(
            s_ctx, f_ctx, valid_lens_ctx
        )
        r_ctx_boot = self.encode_deterministic(
            s_ctx_boot, f_ctx_boot, valid_lens_ctx_boot, training
        )
        output_boot = self.decode(
            rep(r_ctx),
            rep(s_ctx),
            rep(s_test),
            valid_lens_ctx_boot,
            training,
            r_ctx_boot,
        )
        output = self.decode(r_ctx, s_ctx, s_test, valid_lens_ctx, training)
        return output_boot, output

    def sample_with_replacement(
        self,
        s_ctx: jax.Array,
        f_ctx: jax.Array,
        valid_lens_ctx: Optional[jax.Array] = None,
    ):
        (B, L, _), K = s_ctx.shape, self.num_samples
        if valid_lens_ctx is None:
            valid_lens_ctx = jnp.repeat(L, B)
        # turn off gradients
        s_ctx, f_ctx, valid_lens_ctx = (
            no_grad(s_ctx),
            no_grad(f_ctx),
            no_grad(valid_lens_ctx),
        )
        # bootstrap sample residuals
        rng_ctx_boot, rng_res_boot = random.split(self.make_rng("extra"))
        rep = jit(lambda x: jnp.repeat(x, K, axis=0))
        s_ctx_boot, valid_lens_ctx_boot = bootstrap(
            rng_ctx_boot, s_ctx, valid_lens_ctx, K
        )
        f_ctx_boot, valid_lens_ctx_boot = bootstrap(
            rng_ctx_boot, f_ctx, valid_lens_ctx, K
        )
        r_ctx_boot = self.encode_deterministic(
            s_ctx_boot, f_ctx_boot, valid_lens_ctx_boot
        )
        s_ctx_rep = rep(s_ctx)
        f_ctx_mu_boot, f_ctx_std_boot = self.decode(
            r_ctx_boot, s_ctx_rep, s_ctx_rep, valid_lens_ctx_boot
        )
        res = (rep(f_ctx) - f_ctx_mu_boot) / f_ctx_std_boot
        res_boot, _ = bootstrap(rng_res_boot, res, valid_lens_ctx_boot)
        mask = mask_from_valid_lens(L, valid_lens_ctx_boot)
        res_boot -= res_boot.mean(axis=1, where=mask, keepdims=True)
        return (
            s_ctx_rep,
            f_ctx_mu_boot + f_ctx_std_boot * res_boot,
            valid_lens_ctx_boot,
        )

    def encode_deterministic(
        self,
        s: jax.Array,  # [B, L, D_s]
        f: jax.Array,  # [B, L, D_f]
        valid_lens: Optional[jax.Array] = None,  # [B]
        training: bool = False,
    ):
        s_f = jnp.concatenate([s, f], -1)
        s_f_embed = self.enc_det(s_f, training)
        r_ctx, _ = self.self_attn_det(
            s_f_embed,
            s_f_embed,
            s_f_embed,
            valid_lens,
            training,
        )
        return r_ctx

    def decode(
        self,
        r_ctx: jax.Array,
        s_ctx: jax.Array,
        s_test: jax.Array,
        valid_lens_ctx: Optional[jax.Array] = None,
        training: bool = False,
        r_ctx_boot: Optional[jax.Array] = None,
    ):
        s_ctx_embed = self.embed_s(s_ctx)
        s_test_embed = self.embed_s(s_test)
        r, _ = self.cross_attn(
            s_test_embed,  # qs
            s_ctx_embed,  # ks
            r_ctx,  # vs
            valid_lens_ctx,
            training,
        )  # [B*K, L_test, d_ffn]
        q = jnp.concatenate([r, s_test], -1)  # [B*K, L_test, d_ffn + D_s]
        h = self.dec_hid(q, training)
        if r_ctx_boot is not None:
            r_boot, _ = self.cross_attn(
                s_test_embed,  # qs
                s_ctx_embed,  # ks
                r_ctx_boot,  # vs
                valid_lens_ctx,
                training,
            )  # [B*K, L_test, d_ffn]
            h += self.dec_boot(r_boot, training)
        f_dist = self.dec_dist(h, training)
        return self.output_fn(f_dist)  # [B*K, L_test, d_f]
