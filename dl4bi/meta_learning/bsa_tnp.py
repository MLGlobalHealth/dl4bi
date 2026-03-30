from collections.abc import Callable
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..core.mlp import MLP
from ..core.model_output import DiagonalMVNOutput
from ..core.transformer import KRBlock
from ..core.utils import safe_stack
from .steps import likelihood_train_step, likelihood_valid_step
from .utils import first_shape


# TODO(danj): make generic to input arguments beyond (x, s, t)
class BSATNP(nn.Module):
    """Biased Scan Attention Transformer Neural Process (BSA-TNP).

    Args:
        num_blks: Number of `KRBlocks` to use.
        num_reps: Number of times to repeat each `KRBlock`.
        embed_s: A module that embeds the index set prior to embedding with
            function values.
        embed_f: A module that embeds function values prior to embedding with
            the index set.
        embed_obs: A module that creates embeddings for observed (context) and
            unobserved (test) points.
        embed_all: A module that jointly embeds `obs`, `s`, and `f` embeddings.
        x_sim: Similarity function for fixed effects. Non-finite values are masked.
        s_sim: Similarity function for spatial effects. Non-finite values are masked.
        t_sim: Similarity function for temporal effects. Non-finite values are masked.
        x_bias: Bias module for fixed similarity values.
        s_bias: Bias module for spatial similarity values.
        t_bias: Bias module for temporal similarity values.
        blk: A block to use for each layer and each repetition.
        norm: A module used for normalization between blocks.
        head: Transforms the tokens into model output.
        output_fn: A function that transforms the model output into
            a form that can be consumed by loss functions.
        train_step: What training step to use.
        valid_step: What validation step to use.

    Returns:
        An instance of the `BSA-TNP` model.
    """

    num_blks: int = 6
    num_reps: int = 1
    embed_x: Callable = lambda x: x
    embed_s: Callable = lambda x: x
    embed_t: Callable = lambda x: x
    embed_f: Callable = lambda x: x
    embed_obs: nn.Module = nn.Embed(2, 4)
    embed_all: nn.Module = MLP([256, 128, 64], nn.gelu)
    blk: nn.Module = KRBlock()
    norm: nn.Module = nn.LayerNorm()
    head: nn.Module = MLP([256, 64, 2], nn.gelu)
    output_fn: Callable = DiagonalMVNOutput.from_activations
    train_step: Callable = likelihood_train_step
    valid_step: Callable = likelihood_valid_step

    @nn.compact
    def __call__(
        self,
        x_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_x]
        s_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_s]
        t_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_t]
        f_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_f]
        mask_ctx: Optional[jax.Array] = None,  # [B, L_ctx]
        x_test: Optional[jax.Array] = None,  # [B, L_test, D_x]
        s_test: Optional[jax.Array] = None,  # [B, L_test, D_s]
        t_test: Optional[jax.Array] = None,  # [B, L_test, D_t]
        training: bool = False,
        **kwargs,
    ):
        r"""Run module forward.

        Args:
            rng: A psuedo-random number generator.
            x_ctx: Optional fixed effects for context points.
            t_ctx: Optional temporal values for context points.
            s_ctx: Optional spatial values for context points.
            f_ctx: Function values for context points.
            mask_ctx: A mask for context points.
            x_test: Optional fixed effects for test points.
            t_test: Optional temporal values for test points.
            s_test: Optional spatial values for test points.
            f_test: Function values for test points.
            training: A boolean indicating whether this call is performed during
                training.

        Returns:
            `ModelOutput`.
        """
        test_shape = first_shape([x_test, s_test, t_test])
        f_test = jnp.zeros((*test_shape[:-1], f_ctx.shape[-1]))
        obs = jnp.ones(f_ctx.shape[:-1], dtype=jnp.uint8)
        unobs = jnp.zeros(f_test.shape[:-1], dtype=jnp.uint8)
        ctx = safe_stack(
            self.embed_obs(obs),
            self.embed_x(x_ctx),
            self.embed_s(s_ctx),
            self.embed_t(t_ctx),
            self.embed_f(f_ctx),
        )
        test = safe_stack(
            self.embed_obs(unobs),
            self.embed_x(x_test),
            self.embed_s(s_test),
            self.embed_t(t_test),
            self.embed_f(f_test),
        )
        norm = self.norm.copy()
        qvs, kvs = map(lambda x: norm(self.embed_all(x)), (test, ctx))
        qk_kwargs = dict(
            qs_x=x_test,
            qs_s=s_test,
            qs_t=t_test,
            ks_x=x_ctx,
            ks_s=s_ctx,
            ks_t=t_ctx,
        )
        kk_kwargs = dict(
            qs_x=x_ctx,
            qs_s=s_ctx,
            qs_t=t_ctx,
            ks_x=x_ctx,
            ks_s=s_ctx,
            ks_t=t_ctx,
        )
        for _ in range(self.num_blks):
            blk = self.blk.copy()
            for _ in range(self.num_reps):
                qvs, kvs = blk(qvs, kvs, mask_ctx, training, qk_kwargs, kk_kwargs)
        qvs = self.norm.copy()(qvs)
        output = self.head(qvs, training)
        return self.output_fn(output)
