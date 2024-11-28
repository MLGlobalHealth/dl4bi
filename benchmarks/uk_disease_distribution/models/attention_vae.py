#!/usr/bin/env python3
import jax.numpy as jnp
from flax import linen as nn
from jax import Array, random, vmap
from sps.kernels import l2_dist

from dl4bi.core import MLP, TransformerEncoder, TransformerEncoderBlock


class AttentionVAE(nn.Module):
    feat_enc: nn.Module = MLP([64] * 2)
    blk: nn.Module = TransformerEncoder(
        num_blks=3,
        blk=TransformerEncoderBlock(
            ffn=MLP([64, 64], nn.relu),
        ),
    )
    head_z_mu: nn.Module = MLP([64, 1], nn.relu)
    head_z_log_var: nn.Module = MLP([64, 1], nn.relu)
    condition_bias: bool = True

    @nn.compact
    def __call__(self, f, conditionals, **kwargs):
        if self.condition_bias and kwargs.get("bias", None) is None:
            raise ValueError("Must specify a bias to condition on!")
        if self.condition_bias:
            bias = kwargs["bias"]
            broad_cond = jnp.broadcast_to(
                conditionals, bias.shape + (len(conditionals),)
            )
            kwargs["bias"] = MLP([1])(jnp.concatenate([broad_cond, bias], axis=-1))
        else:
            broad_cond = jnp.broadcast_to(conditionals, f.shape + (len(conditionals),))
            f = jnp.concatenate([f, broad_cond], axis=-1)
        f = self.feat_enc(f)
        z = self.blk(f, None, False, **kwargs)
        z_mu = self.head_z_mu(z)
        z_log_var = self.head_z_log_var(z)
        z_std = jnp.exp(z_log_var / 2)
        return z_mu, z_std


class TransformerVAE(nn.Module):
    r"""Transformer based VAE.

    Once trained, the module's `decoder` can be used as a generative
    model to simulate a GP from the samples it was trained on.

    Args:
        encoder: A module used to encode GP realizations and
            their hyperparamters.
        decoder: A module used to decode random vectors and
            GP hyperparameters into GP samples.
        z_dim: The size of the hidden dimension.

    Returns:
        An instance of the PriorCVAE network.
        $\hat{\mathbf{f}}$, a recreation of the original $\mathbf{f}$,
        along with $\mu$ and $\log(\sigma^2)$, which are often used
        to calculate losses involving KL divergence.
    """

    z_dim: int = 363
    condition_bias: bool = True
    bias_attention: bool = True

    @nn.compact
    def __call__(self, f: Array, conditionals: list[Array], **kwargs):
        r"""Run module forward.

        Args:
            f: The function values, an array of shape `(B, K, 1)`.
            conditionals: list of conditional variavles to condition the
            output on. NOTE: Must ensure same ordering of conditional
            at all times, as the model implicitly learns this order.

        Returns:
            $\hat{\mathbf{f}}$, a recreation of the original$\mathbf{f}$,
            along with $\mu$ and $\log(\sigma^2)$, which are often used
            to calculate losses involving KL divergence.
        """
        encoder: nn.Module = AttentionVAE(condition_bias=self.condition_bias)
        decoder: nn.Module = AttentionVAE(condition_bias=self.condition_bias)
        z_mu, z_std = encoder(f, conditionals, **kwargs)
        eps = random.normal(self.make_rng("extra"), z_std.shape)
        z = z_mu + z_std * eps
        f_hat, _ = decoder(z, conditionals, **kwargs)
        return f_hat.reshape(f.shape), z_mu, z_std
