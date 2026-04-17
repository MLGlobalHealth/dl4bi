from collections.abc import Callable
from typing import Optional, Union

import flax.linen as nn
import jax.numpy as jnp
from jax import Array

from ..core.attention import MultiHeadAttention
from ..core.bias import Bias
from ..core.mlp import MLP, gMLP, gMLPBlock
from ..core.model_output import VAEOutput
from ..core.transformer import TransformerEncoderBlock
from ..vae.train_utils import cond_as_feats, cond_as_locs


class DeepRV(nn.Module):
    r"""`DeepRV` learns to emulate samples from a fixed size stochastic process.

    The model learns the function $f_\mathbf{c}:(\mathbf{z})\to\mathbf{T}_{\mathbf{c}}\mathbf{z}$,
    from the latent space to the realizations of the process given conditional hyperparameters $\mathbf{c}$,
    where $\mathbf{L}_{\mathbf{c}}$ is conditioned on the hyperparameters $\mathbf{c}$.

    E.g for a Gaussian Process (GP) with a kernel $\mathcal{K}_\mathbf{c}$, and a fixed spatial structure
    $\mathbf{x}$, the model emulates the GP by learning the following function:
    $\text{Cholesky}(\mathbf{K}_\mathbf{c})\mathbf{z} \approx  \text{DeepRV}(\mathbf{z}, \mathbf{c})$,
    which subsequentially gives us:

    $\mathbf{f}_\mathbf{c}\sim\mathcal{GP}_\mathbf{c}(\cdot,\cdot) \approx \mathbf{\hat{f}}_{\mathbf{c}} =
    \text{DeepRV}(\mathbf{z}, \mathbf{c}), \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$

    The decoder submodule can be any neural network whose output
    size is `num_locations`.

    Args:
        z: latent vector samples. The first dimension is assumed to be the
            batch dimension.
        conditionals: conditional hyperparameters of the process.

    Returns:
        An instance of the `DeepRV` network.
    """

    decoder: nn.Module
    cond_stack_fn: Callable

    @nn.compact
    def __call__(self, z: Array, conditionals: Array, **kwargs):
        r"""Run module forward.

        Args:
            z: latent vector samples.
            conditionals: conditional hyperparameters of the process.

        Returns:
            $\hat{\mathbf{f}}$, an approximation of the stochastic process's realizations.
        """
        return VAEOutput(self.decode(z, conditionals, **kwargs))

    def decode(self, z: Array, conditionals: Array, **kwargs):
        return self.decoder(self.cond_stack_fn(z, conditionals), **kwargs)


# NOTE: Explicit decoder definitions for DeepRV
class gMLPDeepRV(nn.Module):
    num_blks: int = 2
    s_embed: Union[Callable, nn.Module] = lambda s: s
    proj_in: nn.Module = MLP([128, 128], nn.gelu)
    proj_out: nn.Module = MLP([64, 64], nn.gelu)
    attn: Optional[nn.Module] = None
    gate_fn: Union[Callable, nn.Module] = lambda x: x
    embed: nn.Module = MLP([64, 64], nn.gelu)
    head: nn.Module = MLP([128, 1], nn.gelu)

    @nn.compact
    def __call__(self, z: Array, conditionals: Array, s: Array, **kwargs):
        s_embeded = self.s_embed(s)
        batched_s = jnp.repeat(s_embeded[None, ...], z.shape[0], axis=0)
        x = jnp.concat([jnp.atleast_3d(z), batched_s], axis=-1)
        x = cond_as_feats(x, conditionals)
        return VAEOutput(
            gMLP(
                num_blks=self.num_blks,
                embed=self.embed,
                blk=gMLPBlock(self.proj_in, self.proj_out, self.attn, self.gate_fn),
                head=self.head,
            )(x, **kwargs)
        )

    def decode(self, z: Array, conditionals: Array, s: Array, **kwargs):
        return self(z, conditionals, s, **kwargs).f_hat


class RecursiveGMLPDeepRV(nn.Module):
    """DeepRV variant that iteratively refines a shared gMLP state."""

    num_cycles: int = 3
    s_embed: Union[Callable, nn.Module] = lambda s: s
    proj_in: nn.Module = MLP([128, 128], nn.gelu)
    proj_out: nn.Module = MLP([64, 64], nn.gelu)
    attn: Optional[nn.Module] = None
    gate_fn: Union[Callable, nn.Module] = lambda x: x
    embed: nn.Module = MLP([64, 64], nn.gelu)
    head: nn.Module = MLP([128, 1], nn.gelu)

    @nn.compact
    def __call__(
        self,
        z: Array,
        conditionals: Array,
        s: Array,
        num_cycles: Optional[int] = None,
        **kwargs,
    ):
        s_embeded = self.s_embed(s)
        batched_s = jnp.repeat(s_embeded[None, ...], z.shape[0], axis=0)
        x = jnp.concat([jnp.atleast_3d(z), batched_s], axis=-1)
        x = cond_as_feats(x, conditionals)
        x = self.embed(x)
        block = gMLPBlock(
            self.proj_in,
            self.proj_out,
            attn=self.attn,
            gate_fn=self.gate_fn,
        )
        norm = nn.LayerNorm()
        cycles = self.num_cycles if num_cycles is None else num_cycles
        preds = []
        for _ in range(cycles):
            x = x + block(norm(x), **kwargs)
            preds.append(self.head(norm(x)).squeeze(-1))
        return VAEOutput(jnp.stack(preds, axis=0))

    def decode(
        self,
        z: Array,
        conditionals: Array,
        s: Array,
        num_cycles: Optional[int] = None,
        **kwargs,
    ):
        return self(z, conditionals, s, num_cycles=num_cycles, **kwargs).f_hat


class MLPDeepRV(nn.Module):
    dims: list[int]

    @nn.compact
    def __call__(self, z: Array, conditionals: Array, **kwargs):
        x = cond_as_locs(z, conditionals)
        return VAEOutput(MLP(self.dims)(x))

    def decode(self, z: Array, conditionals: Array, **kwargs):
        return self(z, conditionals, **kwargs).f_hat


class KernelBiasTransformerDeepRV(nn.Module):
    max_locations: int
    dim: int = 64
    num_blks: int = 2
    s_embed: Union[Callable, nn.Module] = lambda x: x
    head: Union[Callable, nn.Module] = MLP([128, 1], nn.gelu)

    @nn.compact
    def __call__(
        self,
        z: Array,
        conditionals: Array,
        s: Array,
        K: Array,
        mask: Optional[Array] = None,
        **kwargs,
    ):
        (B, L), D, C = z.shape, self.dim, conditionals.shape[0]
        batched_s = jnp.repeat(s[None, ...], z.shape[0], axis=0)
        s_embeded = self.s_embed(batched_s)
        ids = jnp.repeat(jnp.arange(L, dtype=int)[None, :], B, axis=0)
        ids_embed = nn.Embed(self.max_locations, features=(D * 2) - (C + 1))(ids)
        x = jnp.concat([jnp.atleast_3d(z), s_embeded, ids_embed], axis=-1)
        x = cond_as_feats(x, conditionals)
        x = MLP([D * 4, D], nn.gelu)(x)
        for _ in range(self.num_blks):
            kwargs = {"bias": Bias.build_scalar_bias()(jnp.repeat(K[None], B, axis=0))}
            attn = MultiHeadAttention(
                proj_qs=MLP([D * 2]),
                proj_ks=MLP([D * 2]),
                proj_vs=MLP([D * 2]),
                proj_out=MLP([D]),
            )
            ffn = MLP([D * 4, D])
            x, _ = TransformerEncoderBlock(attn=attn, ffn=ffn)(
                x, mask=mask, training=False, **kwargs
            )
        return VAEOutput(self.head(x))

    def decode(self, z: Array, conditionals: Array, **kwargs):
        return self(z, conditionals, **kwargs).f_hat


class FixedKernelAttention(nn.Module):
    proj_vs: nn.Module = MLP([64], nn.gelu)

    @nn.compact
    def __call__(self, qs, ks, vs, valid_lens: Optional[Array] = None, **kwargs):
        attn_scores = nn.softmax(kwargs["K"], axis=-1)
        attn_res = jnp.einsum("ij,bjd->bid", attn_scores, self.proj_vs(vs))
        attn_w = self.param("kernel_attn_scale", lambda k: jnp.array(0.1))
        attn_res = attn_w.clip(0.0, 10.0) * attn_res
        return attn_res, attn_scores
