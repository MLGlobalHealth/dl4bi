from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import jit, random
from jax.lax import stop_gradient as no_grad

from ..core import MLP, bootstrap, mask_from_valid_lens


class BNP(nn.Module):
    """The Bootstrapping Neural Process as detailed in [Bootstrapping Neural Processes](https://arxiv.org/abs/2008.02956).

    This implementation is based on the official implementation
    [here](https://github.com/juho-lee/bnp/tree/master), although
    we use the hyperparameters specified in Figure 8 on page 12 of
    [Attentive Neural Processes](https://arxiv.org/abs/1901.05761)
    to keep comparisons among models consistent.

    Args:
        enc_det: A module for encoding context points.
        dec_hid: The first stage of decoding at test points.
        dec_boot: A decoding module that integrates bootstrapped samples.
        dec_dist: Decodes the hidden state into a distribution at test points.
        num_samples: The number of samples to use for bootstrapping.
        min_std: Bounds standard deviation, default 0.0 (original 0.1).

    Returns:
        An instance of `BNP`.
    """

    enc_det: nn.Module = MLP([128] * 6)
    dec_hid: nn.Module = MLP([128])
    dec_boot: nn.Module = MLP([128] * 2)
    dec_dist: nn.Module = MLP([128] * 3 + [2])
    num_samples: int = 4
    min_std: float = 0.0

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
        f_mu_boot, f_std_boot = self.decode(
            rep(r_ctx), rep(s_test), training, r_ctx_boot
        )
        f_mu, f_std = self.decode(r_ctx, s_test, training)
        return f_mu_boot, f_std_boot, f_mu, f_std

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
        f_ctx_mu_boot, f_ctx_std_boot = self.decode(r_ctx_boot, s_ctx_rep)
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
        s_ctx: jax.Array,  # [B, L, D_s]
        f_ctx: jax.Array,  # [B, L, D_f]
        valid_lens_ctx: Optional[jax.Array] = None,  # [B]
        training: bool = False,
    ):
        (B, L, _) = s_ctx.shape
        if valid_lens_ctx is None:
            valid_lens_ctx = jnp.repeat(L, B)
        mask = mask_from_valid_lens(L, valid_lens_ctx)
        s_f_ctx = jnp.concatenate([s_ctx, f_ctx], -1)
        s_f_ctx_embed = self.enc_det(s_f_ctx, training)
        return jnp.mean(s_f_ctx_embed, axis=1, where=mask)  # [B, d_ffn]

    def decode(
        self,
        r_ctx: jax.Array,
        s_test: jax.Array,
        training: bool = False,
        r_ctx_boot: Optional[jax.Array] = None,
    ):
        L_test = s_test.shape[1]
        r_ctx = jnp.repeat(r_ctx[:, None, :], L_test, axis=1)  # [B*K, L_test, d_ffn]
        q = jnp.concatenate([r_ctx, s_test], -1)  # [B*K, L_test, d_ffn + D_s]
        h = self.dec_hid(q, training)
        if r_ctx_boot is not None:
            r_ctx_boot = jnp.repeat(r_ctx_boot[:, None, :], L_test, axis=1)
            h += self.dec_boot(r_ctx_boot, training)
        f_dist = self.dec_dist(h, training)
        f_mu, f_std = jnp.split(f_dist, 2, axis=-1)
        f_std = self.min_std + (1 - self.min_std) * nn.softplus(f_std)
        return f_mu, f_std  # [B*K, L_test, d_f]
