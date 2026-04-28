from collections.abc import Callable
from typing import Optional, Union

import flax.linen as nn
import jax.numpy as jnp
from jax import Array

from ..core.mlp import MLP, gMLP, gMLPBlock
from ..core.model_output import VAEOutput
from ..vae.train_utils import cond_as_feats


class FlowMatchingVectorField(nn.Module):
    r"""gMLP vector field network for conditional flow matching.

    Mirrors ``gMLPDeepRV`` but accepts per-sample time ``t ∈ [0, 1]``
    as an additional conditioning feature tiled across locations.

    Args:
        num_blks: Number of gMLP blocks.
        s_embed: Spatial coordinate embedding (defaults to identity).
        proj_in: Projection applied inside each gMLP block.
        proj_out: Projection applied after gating inside each block.
        attn: Optional attention module for aMLP variant.
        gate_fn: Gating activation (defaults to identity).
        embed: Embedding applied before the gMLP stack.
        head: Final projection to a scalar per location.
    """

    num_blks: int = 2
    s_embed: Union[Callable, nn.Module] = lambda s: s
    proj_in: nn.Module = MLP([128, 128], nn.gelu)
    proj_out: nn.Module = MLP([64, 64], nn.gelu)
    attn: Optional[nn.Module] = None
    gate_fn: Union[Callable, nn.Module] = lambda x: x
    embed: nn.Module = MLP([64, 64], nn.gelu)
    head: nn.Module = MLP([128, 1], nn.gelu)

    @nn.compact
    def __call__(
        self, x_t: Array, conditionals: Array, t: Array, s: Array, **kwargs
    ) -> VAEOutput:
        r"""Predict the vector field :math:`v_\theta(x_t, t, \mathbf{c}, s)`.

        Args:
            x_t: Interpolated sample ``[B, L]``.
            conditionals: Kernel hyperparameters ``[C]``, shared across batch.
            t: Per-sample time values ``[B]``.
            s: Spatial coordinates ``[L, d]``.

        Returns:
            ``VAEOutput`` whose ``f_hat`` is the predicted vector field
            of shape ``[B, L, 1]``.
        """
        s_embedded = self.s_embed(s)
        batched_s = jnp.repeat(s_embedded[None, ...], x_t.shape[0], axis=0)
        x = jnp.concat([jnp.atleast_3d(x_t), batched_s], axis=-1)
        x = cond_as_feats(x, conditionals)
        # Per-sample time: broadcast t [B] → [B, L, 1] and append
        t_feat = jnp.repeat(t[:, None, None], x.shape[1], axis=1)
        x = jnp.concat([x, t_feat], axis=-1)
        v = gMLP(
            num_blks=self.num_blks,
            embed=self.embed,
            blk=gMLPBlock(self.proj_in, self.proj_out, self.attn, self.gate_fn),
            head=self.head,
        )(x, **kwargs)
        return VAEOutput(v)


class FlowMatchingDeepRV(nn.Module):
    r"""Flow matching surrogate that replaces the MLP/gMLP decoder in DeepRV.

    Trains a vector field :math:`v_\theta(x_t, t, \mathbf{c})` using the
    conditional flow matching (CFM) objective.  The target flow transports
    :math:`\mathcal{N}(\mathbf{0}, \mathbf{I})` to GP samples along
    straight paths (optimal-transport CFM):

    .. math::
        x_t = (1-t)\,\mathbf{z} + t\,\mathbf{f}, \qquad
        v^*(x_t, t) = \mathbf{f} - \mathbf{z}

    At inference, the ODE :math:`\dot{x} = v_\theta(x, t, \mathbf{c})`
    is integrated from :math:`x_0 = \mathbf{z}` using ``n_steps`` Euler
    steps, yielding a deterministic GP-sample approximation given
    :math:`\mathbf{z}`.

    ``decode`` is a drop-in for ``DeepRV.decode`` and is compatible with
    :func:`~dl4bi.vae.train_utils.generate_surrogate_decoder` and HMC.

    Training:
        Use :func:`~dl4bi.vae.train_utils.flow_matching_train_step` and
        :func:`~dl4bi.vae.train_utils.flow_matching_valid_step`.

    Args:
        vf: Vector field network, e.g. :class:`FlowMatchingVectorField`.
        n_steps: Euler steps at inference.  ``n_steps=1`` is one forward
            pass (same cost as current DeepRV); more steps improve sample
            quality at the cost of inference time.
    """

    vf: nn.Module
    n_steps: int = 1

    @nn.compact
    def __call__(self, z: Array, conditionals: Array, t: Array, **kwargs) -> VAEOutput:
        r"""Predict the vector field at the interpolated sample ``z = x_t``.

        During training ``z`` is the pre-computed interpolation
        :math:`x_t = (1-t)\mathbf{z}_0 + t\mathbf{f}`.
        :func:`~dl4bi.vae.train_utils.flow_matching_train_step` constructs
        ``x_t`` before invoking this method.

        Args:
            z: Interpolated sample ``[B, L]`` (= ``x_t`` during training).
            conditionals: Kernel hyperparameters ``[C]``.
            t: Per-sample time ``[B]``.
            **kwargs: Forwarded to the vector field (e.g. ``s=coords``).

        Returns:
            ``VAEOutput`` with ``f_hat`` = predicted vector field.
        """
        return self.vf(z, conditionals, t, **kwargs)

    def decode(self, z: Array, conditionals: Array, **kwargs) -> Array:
        r"""Integrate the ODE from ``z`` to an approximate GP sample.

        Args:
            z: Initial noise ``[B, L]``, typically sampled by HMC as
               ``z ~ N(0, I)``.
            conditionals: Kernel hyperparameters ``[C]``.
            **kwargs: Forwarded to the vector field (e.g. ``s=coords``).

        Returns:
            Approximate GP sample of shape ``[B, L]``.
        """
        x = z
        dt = 1.0 / self.n_steps
        for i in range(self.n_steps):
            # Midpoint rule: evaluate vector field at centre of each interval
            t = jnp.full((x.shape[0],), (i + 0.5) * dt)
            v = self.vf(x, conditionals, t, **kwargs).f_hat
            x = x + dt * v.reshape(x.shape)
        return x
