from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..core.mlp import MLP
from ..core.model_output import DiagonalMVNOutput
from ..core.transformer import DistillBlock
from ..core.utils import safe_stack
from .steps import likelihood_train_step, likelihood_valid_step
from .utils import first_shape


class SPDistiller(nn.Module):
    """A Stochastic Process distiller for self-supervised learning.

    Args:
        norm: A module used for normalization between blocks.
        head: Transforms the tokens into model output.
        output_fn: A function that transforms the model output into
            a form that can be consumed by loss functions.
        train_step: What training step to use.
        valid_step: What validation step to use.

    Returns:
        An instance of the `HLNP` model.
    """

    embed_x: Callable = lambda x: x
    embed_s: Callable = lambda x: x
    embed_t: Callable = lambda x: x
    embed_f: Callable = lambda x: x
    embed_obs: nn.Module = nn.Embed(2, 4)
    embed_all: nn.Module = MLP([256, 128, 64], nn.gelu)
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
        norm = nn.LayerNorm()
        qs, ks_0 = map(lambda x: norm(self.embed_all(x)), (test, ctx))
        s_grid = build_grid(
            [
                dict(start=lo, stop=up, num=int(self.points_per_unit * (up - lo)))
                for (lo, up) in zip(self.s_lower, self.s_upper)
            ]
        )  # [*P..., s_dim]
        # qs = self.param("latents", init.truncated_normal(), (1, Z, D))
        # qs = jnp.repeat(qs, B, axis=0)
        # TODO(danj): add positional embeddings to distill blocks
        ks_1 = DistillBlock(64)(ks_0, mask_ctx, training)
        ks_2 = DistillBlock(16)(ks_1, None, training)
        ks_3 = DistillBlock(4)(ks_2, None, training)
        qs = self.norm(qs)
        output = self.head(qs, training)
        return self.output_fn(output)
