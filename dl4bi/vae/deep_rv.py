from collections.abc import Callable

import flax.linen as nn
import jax.numpy as jnp
from jax import Array
from sps.kernels import l2_dist

from ..core.attention import (
    BiasedScanAttention,
    DeepKernelAttention,
    MultiHeadAttention,
)
from ..core.bias import Bias
from ..core.mlp import MLP, gMLP
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

    @nn.compact
    def __call__(self, z: Array, conditionals: Array, s: Array, **kwargs):
        batched_s = jnp.repeat(s[None, ...], z.shape[0], axis=0)
        x = jnp.concat([jnp.atleast_3d(z), batched_s], axis=-1)
        x = cond_as_feats(x, conditionals)
        return VAEOutput(gMLP(self.num_blks)(x))

    def decode(self, z: Array, conditionals: Array, s: Array, **kwargs):
        return self(z, conditionals, s, **kwargs).f_hat


class MLPDeepRV(nn.Module):
    dims: list[int]

    @nn.compact
    def __call__(self, z: Array, conditionals: Array, **kwargs):
        x = cond_as_locs(z, conditionals)
        return VAEOutput(MLP(self.dims)(x))

    def decode(self, z: Array, conditionals: Array, **kwargs):
        return self(z, conditionals, **kwargs).f_hat


class TransformerDeepRV(nn.Module):
    dim: int = 64
    num_blks: int = 2

    @nn.compact
    def __call__(self, z: Array, conditionals: Array, s: Array, **kwargs):
        (B, L), D, C = z.shape, self.dim, conditionals.shape[0]
        ids = jnp.repeat(jnp.arange(L, dtype=int)[None, :], B, axis=0)
        ids_embed = nn.Embed(L, features=(D * 2) - (C + 1))(ids)
        x = cond_as_feats(z, conditionals)
        x = jnp.concat([ids_embed, x], axis=-1)
        x = MLP([D * 4, D], nn.gelu)(x)
        # TODO(jhonathan): cache d
        d = jnp.repeat(l2_dist(s, s)[None, ...], B, axis=0)
        for _ in range(self.num_blks):
            attn = MultiHeadAttention(
                proj_qs=MLP([D * 2]),
                proj_ks=MLP([D * 2]),
                proj_vs=MLP([D * 2]),
                proj_out=MLP([D]),
            )
            ffn = MLP([D * 4, D])
            bias = Bias.build_rbf_network_bias()(d)
            x, _ = TransformerEncoderBlock(attn=attn, ffn=ffn)(x, bias=bias)
        return VAEOutput(MLP([D * 4, D, 1])(x))

    def decode(self, z: Array, conditionals: Array, **kwargs):
        return self(z, conditionals, **kwargs).f_hat


class ScanTransformerDeepRV(nn.Module):
    dim: int = 64
    num_blks: int = 2

    @nn.compact
    def __call__(self, z: Array, conditionals: Array, s: Array, **kwargs):
        (B, L), D, C = z.shape, self.dim, conditionals.shape[0]
        s_batched = jnp.repeat(s[None, :], B, axis=0)
        ids = jnp.repeat(jnp.arange(L, dtype=int)[None, :], B, axis=0)
        ids_embed = nn.Embed(L, features=(D * 2) - (C + 1))(ids)
        x = cond_as_feats(z, conditionals)
        x = jnp.concat([ids_embed, x], axis=-1)
        x = MLP([D * 4, D], nn.gelu)(x)
        kwargs = {"qs_s": s_batched, "ks_s": s_batched}
        for _ in range(self.num_blks):
            attn = MultiHeadAttention(
                proj_qs=MLP([D * 2]),
                proj_ks=MLP([D * 2]),
                proj_vs=MLP([D * 2]),
                proj_out=MLP([D]),
                attn=BiasedScanAttention(bias={"s": Bias.build_rbf_network_bias()}),
            )
            ffn = MLP([D * 4, D])
            x, _ = TransformerEncoderBlock(attn=attn, ffn=ffn)(
                x, mask=None, training=False, **kwargs
            )
        return VAEOutput(MLP([D * 4, D, 1])(x))

    def decode(self, z: Array, conditionals: Array, **kwargs):
        return self(z, conditionals, **kwargs).f_hat


class DKADeepRV(nn.Module):
    dim: int = 64
    num_blks: int = 2

    @nn.compact
    def __call__(self, z: Array, conditionals: Array, s: Array, **kwargs):
        (B, L), D, C = z.shape, self.dim, conditionals.shape[0]
        s_batched = jnp.repeat(s[None, :], B, axis=0)
        ids = jnp.repeat(jnp.arange(L, dtype=int)[None, :], B, axis=0)
        ids_embed = nn.Embed(L, features=(D * 2) - (C + 1))(ids)
        x = cond_as_feats(z, conditionals)
        x = jnp.concat([ids_embed, x], axis=-1)
        x = MLP([D * 4, D], nn.gelu)(x)
        kwargs = {"qs_s": s_batched, "ks_s": s_batched}
        for _ in range(self.num_blks):
            attn = DeepKernelAttention(
                num_heads=4,
                proj_qks=MLP([D * 2, D * 2]),
                proj_vs=MLP([D * 2, D]),
            )
            ffn = MLP([D * 4, D])
            x, _ = TransformerEncoderBlock(attn=attn, ffn=ffn)(
                x, mask=None, training=False, **kwargs
            )
        return VAEOutput(MLP([D * 4, D, 1])(x))

    def decode(self, z: Array, conditionals: Array, **kwargs):
        return self(z, conditionals, **kwargs).f_hat
