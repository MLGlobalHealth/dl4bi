from typing import Optional

import jax.numpy as jnp
from flax import linen as nn
from jax import Array, random

from dl4bi.core import MLP, DistanceBias, TransformerEncoderBlock


class AttentionVAE(nn.Module):
    feat_enc: nn.Module
    head: nn.Module
    condition_bias: bool = True
    blk: nn.Module = TransformerEncoderBlock()
    norm: nn.Module = nn.LayerNorm()
    bias_fn: Optional[nn.Module] = DistanceBias()
    num_blks: int = 6

    @nn.compact
    def __call__(self, f: Array, conditionals: list[Array], **kwargs):
        r"""Run module forward.

        Args:
            f: The function values, an array of shape `(B, K, D)`.
            or the latent 'z' if the model is used in decoder only mode.
            conditionals: list of conditional variavles to condition the
            output on. NOTE: Must ensure same ordering of conditional
            at all times, as the model implicitly learns this order.

        Returns:
            $\mathbf{z}_{\mu}$, $\mathbf{z}_{\sigma}$, the
            distribution of the latents, or simply the reconstruction
            if used a decoder.
        """
        if self.condition_bias and kwargs.get("bias", None) is None:
            raise ValueError("Must specify a bias to condition on!")
        B = f.shape[0]
        bias = kwargs.get("bias", None)
        conditionals = jnp.concatenate(conditionals)
        if self.condition_bias:
            broad_cond = jnp.broadcast_to(
                conditionals, bias.shape + (len(conditionals),)
            )
            kwargs["bias"] = MLP([1])(
                jnp.concatenate([broad_cond, bias[..., None]], axis=-1)
            ).squeeze()
        else:
            broad_cond = jnp.broadcast_to(
                conditionals, f.shape[:-1] + (len(conditionals),)
            )
            f = jnp.concatenate([f, broad_cond], axis=-1)
        pos_enc = kwargs.get("pos_enc", None)
        if pos_enc is not None:
            # TODO(jhoot): sign flip with training
            broad_pos_enc = jnp.broadcast_to(pos_enc, (f.shape[0],) + pos_enc.shape)
            f = jnp.concatenate([f, broad_pos_enc], axis=-1)
        f = self.feat_enc(f)
        if bias is not None:
            bias = jnp.repeat(bias[None, ...], B, axis=0)
        for _ in range(self.num_blks):
            if bias is not None and self.bias_fn is not None:
                kwargs["bias"] = self.bias_fn.copy()(bias)
            f, _ = self.blk.copy()(f, None, False, **kwargs)
        if self.blk.pre_norm:
            f = self.norm(f)
        return self.head(f)


class TransformerVAE(nn.Module):
    r"""Transformer based VAE.

    Once trained, the module's `decoder` can be used as a generative
    model to simulate a processses from the samples it was trained on.

    Args:
        encoder: A module used to encode processes and
            their conditional values.
        decoder: A module used to decode normal(0,1) vectors and
            conditionals into their priors.
        z_dim: The size of the hidden dimension (input of the decoder).
        decoder_only: whether to use always only the decoder, for deepChol models
    """

    z_dim: int
    encoder: Optional[nn.Module]
    decoder: nn.Module
    decoder_only: bool = False

    @nn.compact
    def __call__(
        self, f: Array, conditionals: list[Array], decode_only: bool = False, **kwargs
    ):
        r"""Run module forward.

        Args:
            f: The function values, an array of shape `(B, L, D)`.
            or the latent 'z' if the model is used in decoder only mode.
            conditionals: list of conditional variavles to condition the
            output on. NOTE: Must ensure same ordering of conditional
            at all times, as the model implicitly learns this order.

        Returns:
            $\hat{\mathbf{f}}$, a recreation of the original$\mathbf{f}$,
            along with $\mu$ and $\log(\sigma^2)$, which are often used
            to calculate losses involving KL divergence.
        """
        if decode_only or self.decoder_only:
            z, z_mu, z_std = f, None, None
            if len(z.shape) < 3:
                z = z[..., None]
        else:
            z_mu_std = self.encoder(f, conditionals, **kwargs)
            z_mu = z_mu_std[..., 0]
            z_std = z_mu_std[..., 1]
            eps = random.normal(self.make_rng("extra"), z_std.shape)
            z = z_mu + z_std * eps
        f_hat = self.decoder(z, conditionals, **kwargs)
        return f_hat.reshape(f.shape), z_mu, z_std
