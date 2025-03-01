from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import jit, random
from jax.lax import stop_gradient as no_grad

from ..core.mlp import MLP
from ..core.utils import bootstrap
from .model_output import DiagonalMVNOutput


class BNP(nn.Module):
    """The Bootstrapping Neural Process as detailed in [Bootstrapping Neural Processes](https://arxiv.org/abs/2008.02956).

    This implementation is based on the official implementation
    [here](https://github.com/juho-lee/bnp/tree/master), although
    we use the hyperparameters specified in Figure 8 on page 12 of
    [Attentive Neural Processes](https://arxiv.org/abs/1901.05761)
    to keep comparisons among models consistent.

    .. note::
        Currently `BNP` only works with regression.

    Args:
        num_samples: The number of samples to use for bootstrapping.
        enc_det: A module for encoding context points.
        dec_hid: The first stage of decoding at test points.
        dec_boot: A decoding module that integrates bootstrapped samples.
        dec_dist: Decodes the hidden state into model output.
        output_fn: A function that transforms the model output into
            a form that can be consumed by loss functions.

    Returns:
        An instance of `BNP`.
    """

    num_samples: int = 4
    enc_det: nn.Module = MLP([128] * 6)
    dec_hid: nn.Module = MLP([128])
    dec_boot: nn.Module = MLP([128] * 2)
    dec_out: nn.Module = MLP([128] * 3 + [2])
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
        rep = jit(lambda x: jnp.repeat(x, self.num_samples, axis=0))
        r_ctx = self.encode_deterministic(s_ctx, f_ctx, mask_ctx, training)
        s_ctx_boot, f_ctx_boot, mask_ctx_boot = self.sample_with_replacement(
            s_ctx, f_ctx, mask_ctx
        )
        r_ctx_boot = self.encode_deterministic(
            s_ctx_boot, f_ctx_boot, mask_ctx_boot, training
        )
        output_boot = self.decode(rep(r_ctx), rep(s_test), training, r_ctx_boot)
        output = self.decode(r_ctx, s_test, training)
        return output_boot, output

    def sample_with_replacement(
        self,
        s_ctx: jax.Array,
        f_ctx: jax.Array,
        mask_ctx: Optional[jax.Array] = None,
    ):
        K = self.num_samples
        # turn off gradients
        s_ctx, f_ctx, mask_ctx = (
            no_grad(s_ctx),
            no_grad(f_ctx),
            no_grad(mask_ctx),
        )
        # bootstrap sample residuals
        rng_ctx_boot, rng_res_boot = random.split(self.make_rng("extra"))
        rep = jit(lambda x: jnp.repeat(x, K, axis=0))
        s_ctx_boot, mask_ctx_boot = bootstrap(rng_ctx_boot, s_ctx, mask_ctx, K)
        f_ctx_boot, mask_ctx_boot = bootstrap(rng_ctx_boot, f_ctx, mask_ctx, K)
        r_ctx_boot = self.encode_deterministic(s_ctx_boot, f_ctx_boot, mask_ctx_boot)
        s_ctx_rep = rep(s_ctx)
        f_ctx_mu_boot, f_ctx_std_boot = self.decode(r_ctx_boot, s_ctx_rep)
        # TODO(danj): update residual sampling to work with categorical dists
        res = (rep(f_ctx) - f_ctx_mu_boot) / f_ctx_std_boot
        res_boot, _ = bootstrap(rng_res_boot, res, mask_ctx_boot)
        res_boot -= res_boot.mean(axis=1, where=mask_ctx_boot, keepdims=True)
        return s_ctx_rep, f_ctx_mu_boot + f_ctx_std_boot * res_boot, mask_ctx_boot

    def encode_deterministic(
        self,
        s_ctx: jax.Array,  # [B, L_ctx, D_s]
        f_ctx: jax.Array,  # [B, L_ctx, D_f]
        mask_ctx: Optional[jax.Array] = None,  # [B, L_ctx]
        training: bool = False,
    ):
        (B, L, _) = s_ctx.shape
        s_f_ctx = jnp.concatenate([s_ctx, f_ctx], -1)
        s_f_ctx_embed = self.enc_det(s_f_ctx, training)
        if mask_ctx is not None:
            return jnp.mean(s_f_ctx_embed, axis=1, where=mask_ctx[..., None])
        return jnp.mean(s_f_ctx_embed, axis=1)

    def decode(
        self,
        r_ctx: jax.Array,
        s_test: jax.Array,
        training: bool = False,
        r_ctx_boot: Optional[jax.Array] = None,
    ):
        (B, L_test), K = s_test.shape[:2], self.num_samples
        r_ctx = jnp.repeat(r_ctx[:, None, :], L_test, axis=1)  # [B*K, L_test, d_ffn]
        q = jnp.concatenate([r_ctx, s_test], -1)  # [B*K, L_test, d_ffn + D_s]
        h = self.dec_hid(q, training)
        if r_ctx_boot is not None:
            r_ctx_boot = jnp.repeat(r_ctx_boot[:, None, :], L_test, axis=1)
            h += self.dec_boot(r_ctx_boot, training)
        output = self.dec_out(h, training).reshape(B, K, L_test, -1)
        return self.output_fn(output)  # [B, K, L_test, D_f]
