from collections.abc import Callable
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from .attention import MultiheadAttention
from .embed import FixedSinusoidalEmbedding, LearnableEmbedding
from .mlp import MLP
from .transformer import TransformerEncoder


class AttentiveNeuralProcess(nn.Module):
    """(Latent) [Attentive Neural Process](https://arxiv.org/abs/1901.05761).

    Args:
        embed_s: An embedding module for locations.
        embed_s_f: An embedding module for locations and function values.
        enc_ctx_local: An encoder for local context from observed points.
        enc_ctx_global: An encoder for global context from observed points.
        cross_attn: A cross-attention module for matching context and test points.
        pool_func: Callable that pools global context vectors.
        dec_z_mu: A decoder for the mean of latent `z`s.
        dec_z_log_var: A decoder for the log variance of latent `z`s.
        dec_f_mu: A decoder for the mean of the output `f`s.
        dec_f_log_var: A decoder for the log variance of output `f`s.
        num_mc_samples: Number of Monte Carlo samples for each task
            in batch to be used in maximum likelihood loss.

    Returns:
        An instance of the `AttentiveNeuralProcess` model.

    .. warning::
        `valid_lens` applies only to input context sequences, `s_ctx`. Test
        locations, `s_test`, are expected to be dense, i.e. not ragged or
        padded.
    """

    embed_s: nn.Module = LearnableEmbedding(
        FixedSinusoidalEmbedding(128), MLP([128 * 2, 128])
    )
    embed_s_and_f: nn.Module = LearnableEmbedding(
        FixedSinusoidalEmbedding(128), MLP([128 * 2, 128])
    )
    enc_ctx_local: nn.Module = TransformerEncoder()
    enc_ctx_global: nn.Module = TransformerEncoder()
    cross_attn: nn.Module = MultiheadAttention()
    pool_func: Callable = jnp.mean
    dec_z_mu: nn.Module = MLP([128 * 2, 128], p_dropout=0.0)
    dec_z_log_var: nn.Module = MLP([128 * 2, 128], p_dropout=0.0)
    dec_f_mu: nn.Module = MLP([128 * 3, 128 * 2, 128, 1], p_dropout=0.0)
    dec_f_log_var: nn.Module = MLP([128 * 3, 128 * 2, 128, 1], p_dropout=0.0)
    num_mc_samples: int = 32

    @nn.compact
    def __call__(
        self,
        rng: jax.Array,
        s_ctx: jax.Array,  # [B, S_ctx, D_S]
        f_ctx: jax.Array,  # [B, S_ctx, D_F]
        s_test: jax.Array,  # [B, S_test, D_S]
        valid_lens: Optional[jax.Array] = None,  # [B] or [B, S_ctx]
        training: bool = False,
    ):
        r"""Run module forward.

        Args:
            rng: A psuedo-random number generator.
            s_ctx: A location array of shape `(B,S_ctx,D_S)` where
                `B` is batch size, `S_ctx` is number of context
                locations, and `D_S` is the dimension of each location.
            f_ctx: A function value array of shape `(B,S_ctx,D_F)` where `B` is
                batch size, `S_ctx` is number of context locations, and `D_F` is
                the dimension of each function value.
            s_test: A location array of shape `(B,S_test,D_S)` where `B` is
                batch size, `S_test` is number of test locations, and `D_S`
                is the dimension of each location.
            valid_lens: An optional array of shape `(B,)` indicating the
                valid positions for each `S_ctx` sequence in the batch.
            training: A boolean indicating whether this call is performed during
                training.

        Returns:
            $\mu_f,\log(\sigma_f^2)\in\mathbb{R}^{B\times L\times
            S_\text{test}\times D_F}$ and $\mu_z,\log(\sigma_z^2)\in\mathbb{R}
            ^{B\times S_\text{test}\times D_Z}$.
        """
        (B, S_ctx, _), S_test = s_ctx.shape, s_test.shape[1]
        L, mask = self.num_mc_samples, None
        # embed (s,) and (s, f)
        qs, ks = self.embed_s(s_test, training), self.embed_s(s_ctx, training)
        s_and_f_ctx = jnp.concatenate([s_ctx, f_ctx], -1)
        s_and_f_ctx_embed = self.embed_s_and_f(s_and_f_ctx, training)
        # local ("deterministic") path
        vs_local = self.enc_ctx_local(s_and_f_ctx_embed, valid_lens, training)
        rs_local, _ = self.cross_attn(qs, ks, vs_local, valid_lens, training)
        # global ("latent") path
        vs_global = self.enc_ctx_global(s_and_f_ctx_embed, valid_lens, training)
        if isinstance(valid_lens, jax.Array):  # only pool valid values
            mask = (jnp.arange(S_ctx)[None, :] < valid_lens[:, None])[..., None]
        vs_global_pooled = self.pool_func(vs_global, axis=1, where=mask)
        # decode z distribution and sample it num_mc_samples (L) times
        z_mu = self.dec_z_mu(vs_global_pooled, training)  # [B, D_Z]
        z_log_var = self.dec_z_log_var(vs_global_pooled, training)  # [B, D_Z]
        eps = random.normal(rng, (B, L, z_mu.shape[-1]))  # [B, L, D_Z]
        zs_global = z_mu[:, None, :] + jnp.exp(z_log_var / 2)[:, None, :] * eps
        # decode context to (mu_f, log_var_f) for every z sample for every test location
        zs_global_r = jnp.repeat(zs_global[:, :, None, :], S_test, axis=2)
        qs_r = jnp.repeat(qs[:, None, :, :], L, axis=1)
        rs_local_r = jnp.repeat(rs_local[:, None, :, :], L, axis=1)
        ctx = jnp.concatenate([qs_r, rs_local_r, zs_global_r], -1)
        f_mu = self.dec_f_mu(ctx, training)  # [B, L, S_test, D_F]
        f_log_var = self.dec_f_log_var(ctx, training)  # [B, L, S_test, D_F]
        return f_mu, f_log_var, z_mu, z_log_var


# TODO(danj): add importance sampling
def neural_process_maximum_likelihood_loss(
    f_test: jax.Array,  # [B, S_test, D_F]
    f_mu: jax.Array,  # [B, L, S_test, D_F]
    f_log_var: jax.Array,  # [B, L, S_test, D_F]
):
    r"""Maximum likelihood loss from [Meta-Learning Stationary Stochastic Process
    Prediction with Convolutional Neural Processes](https:// arxiv.org/
    abs/2007.01332).

    $$\hat{\mathcal{L}}_\text{ML}(\theta,\phi;\xi)=\log\left[\frac{1}{L}\sum_{l=1}^L\exp\left(\sum_{(\mathbf{s},\mathbf{f})\in D_t}\log p_\theta(\mathbf{f}\mid\mathbf{s},\mathbf{z}_l)\right)\right];\quad\mathbf{z}_l\sim\text{E}_\phi(D_c)$$

    Args:
        f_test: True test function values.
        f_mu: Mean of predicted function values of shape `(B, L, S_test, D_F)`.
        f_log_var: Log variance of predicted function values of shape `(B, L, S_test, D_F)`.

    Returns:
        Maximum likelihood loss for each task averaged over the batch.
    """
    L = f_mu.shape[1]
    f_test = jnp.repeat(f_test[:, None, :, :], L, axis=1)
    logp = norm.logpdf(f_test, f_mu, jnp.exp(f_log_var / 2))
    task_logp_per_sample = jnp.einsum("blsd->bl", logp)
    log_task_logp_mean = logsumexp(task_logp_per_sample, axis=-1) - jnp.log(L)
    return -log_task_logp_mean.mean()  # average across "tasks" in batch
