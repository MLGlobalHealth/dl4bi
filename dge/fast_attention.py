"""
Fast Attention with Orthogonal Random Features ([FAVOR+](https://arxiv.org/abs/2009.14794)).

Implementation based on [google-research fast attention](https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py) and [Teddy Koker's implementation](https://github.com/teddykoker/performer).
"""

from collections.abc import Callable
from functools import partial
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import jit, lax, random, vmap


def gaussian_orf(key: jax.Array, m: int, d: int, structured: bool = True):
    r"""Generates Gaussian [Orthogonal Random Features (ORF)](https://arxiv.org/abs/1610.09072).

    These features are used by [FAVOR+](https://arxiv.org/abs/2009.14794).

    Args:
        rng: A pseudo-random number generator.
        m: The number of rows for ORF matrix.
        d: The number of columns for ORF matrix.
        structured: Whether to use a structured approximation, i.e. SORF.

    Returns:
        A (structured) orthogonal random feature matrix.
    """

    def gaussian_orf_square(rng):
        q, _ = jnp.linalg.qr(random.normal(rng, (d, d)))
        return q.T

    num_squares = int(m / d) + 1
    *rngs, key = random.split(key, num_squares + 1)
    b = lax.map(gaussian_orf_square, jnp.array(rngs)).reshape(-1, d)
    b = b[:m]  # only take first m rows
    if structured:
        multiplier = jnp.sqrt(d) * jnp.ones(m)
    else:  # sample from xi-distribution
        multiplier = jnp.linalg.norm(random.normal(key, (m, d)), axis=1)
    return jnp.diag(multiplier) @ b


def build_simple_positive_softmax_phi(proj: jax.Array):
    r"""Builds the positive softmax kernel from equation (7) in [FAVOR+](https://arxiv.org/abs/2009.14794).

    This version is designed for simplicity, but can be numerically unstable.

    Args:
        proj: A random projection to use for transforming input features.
        eps: An epsilon used for numerical stability.

    Returns:
        $\phi$, a function that maps data to positive vectors used
            in kernel approximation.
    """

    def h(x):
        return jnp.exp(-jnp.square(x).sum(axis=-1, keepdims=True) / 2)

    return build_phi(h, [jnp.exp], proj)


def build_stable_positive_softmax_phi(proj: jax.Array):
    r"""Builds the positive softmax kernel from equation (7) in [FAVOR+](https://arxiv.org/abs/2009.14794).

    This version is optimized for numerical stability.

    Args:
        proj: A random projection to use for transforming input features.

    Returns:
        $\phi$, a function that maps data to positive vectors used
            in kernel approximation.
    """

    @jit
    def phi(x: jax.Array):
        r"""A normalized version of Equation (5) from [FAVOR+](https://arxiv.org/abs/2009.14794).

        $$\frac{\frac{\exp(-\lVert x\rVert^2)}{2}}{\sqrt{m}}\exp(x\Omega)$$
        """
        m, d = proj.shape
        x_proj = jnp.einsum("...d,md->...m", x, proj)
        unexp_h_x = jnp.square(x).sum(-1, keepdims=True) / 2
        # TODO(danj): Google also normalizes by subtracting max value so that
        # exponentiated values lie in [0, 1]. We could do that too, but need to
        # be careful about masked keys. Could use jax.max(..., where=mask).
        return 1 / jnp.sqrt(m) * jnp.exp(x_proj - unexp_h_x)

    return phi


def build_phi(h: Callable, funcs: list[Callable], proj: jax.Array):
    r"""Builds $phi\mathbf{x})$ from equation (5) of [FAVOR+](https://arxiv.org/abs/2009.14794).

    Args:
        h: A coefficient function.
        funcs: The set of mappings to pass random projections through.
        proj: The random projection for input features.

    Returns:
        $\phi$, a function that maps data to vectors used in kernel
            approximation.
    """
    m = proj.shape[0]
    return lambda x: (
        h(x)
        / jnp.sqrt(m)
        * jnp.concatenate(
            [func(jnp.einsum("...d,md->...m", x, proj)) for func in funcs],
            axis=-1,
        )
    )


# TODO(danj): solve numerical stability issue
class FastSoftmaxAttention(nn.Module):
    r"""[FAVOR+](https://arxiv.org/abs/2009.14794) implementation, Appendix B."""

    p_dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, D_QK]
        ks: jax.Array,  # [B, K, D_QK]
        vs: jax.Array,  # [B, K, D_V]
        valid_lens: Optional[jax.Array] = None,  # [B] or [B, Q]
        training: bool = False,
        redraw_random_features: bool = False,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of dimension $\mathbb{R}^{B\times Q\times D_{Q,K}}$
            ks: Keys of dimension $\mathbb{R}^{B\times K\times D_{Q,K}}$
            vs: Values of dimension $\mathbb{R}^{B\times K\tiems D_V}$
            valid_lens: Mask consisting of valid length per sequence of dimension
                $\mathbb{R}^B$.
            training: Boolean indicating whether currently training.
            redraw_random_features: Redraw random features used for
                softmax kernel approximation.

        Returns:
            `ctx` and `attn`, the updated values and None, respectively,
            since the attention matrix is never materialized in FAVOR+.
        """
        B, K, D_QK = ks.shape
        gen_proj = lambda: gaussian_orf(self.make_rng("params"), D_QK, D_QK)
        proj = self.variable("projections", "random", gen_proj)
        if redraw_random_features:
            proj.value = gen_proj()
        normalizer = 1 / jnp.pow(D_QK, 0.25)
        phi = build_stable_positive_softmax_phi(proj.value)
        qs_prime = phi(qs * normalizer)
        ks_prime = phi(ks * normalizer)
        # NOTE: mask after phi in case phi maps zero to non-zero values
        ks_prime = apply_mask(ks_prime, valid_lens)
        ctx = _fast_softmax(qs_prime, ks_prime, vs)
        ctx = nn.Dropout(self.p_dropout, deterministic=not training)(ctx)
        return ctx, None


@jit
def _fast_softmax(
    qs_prime: jax.Array,
    ks_prime: jax.Array,
    vs: jax.Array,
    eps: float = 1e-6,
):
    c = jnp.concatenate([vs, jnp.ones((*vs.shape[:-1], 1))], axis=-1)
    buf_1 = ks_prime.mT @ c
    buf_2 = qs_prime @ buf_1
    buf_3, buf_4 = buf_2[..., :-1], buf_2[..., -1]
    buf_4 = jnp.where(buf_4 < eps, eps, buf_4)  # numerical stabilization
    d_inv = vmap(jnp.diag)(1 / buf_4)
    return d_inv @ buf_3


def apply_mask(x: jax.Array, valid_lens: Optional[jax.Array] = None):
    """Applies a mask using `valid_lens` by zeroing out invalid values."""
    if valid_lens is None:
        return x
    _B, L, _D = x.shape
    mask = (jnp.arange(L) < valid_lens[..., None])[..., None]
    return mask * x


def breakpoint_if_nonfinite(x):
    """Create a breakpoint when non-finite values in `x`."""
    is_finite = jnp.isfinite(x).all()

    def true_fn(x):
        pass

    def false_fn(x):
        jax.debug.breakpoint()

    lax.cond(is_finite, true_fn, false_fn, x)


class MultiheadFastSoftmaxAttention(nn.Module):
    r"""Multihead implementation of [FAVOR+](https://arxiv.org/abs/2009.14794).

    Args:
        num_heads: Number of heads for attention module.
        p_dropout: A dropout rate.

    Returns:
        A `MultiheadFastSoftmaxAttention` module.

    .. note:: This assumes all queries, keys, and values are already embedded, i.e.
        $$
        \begin{aligned}
            \mathbf{Q}&=\mathbf{W}^Q\mathbf{X}\in\mathbb{R}^{N\times D_{Q,K}} \\\\
            \mathbf{K}&=\mathbf{W}^K\mathbf{X}\in\mathbb{R}^{N\times D_{Q,K}} \\\\
            \mathbf{V}&=\mathbf{W}^V\mathbf{Y}\in\mathbb{R}^{N\times D_V} \\\\
        \end{aligned}
        $$
    """

    num_heads: int = 4
    p_dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, D_QK]
        ks: jax.Array,  # [B, K, D_QK]
        vs: jax.Array,  # [B, K, D_V]
        valid_lens: Optional[jax.Array] = None,  # [B] or [B, Q]
        training: bool = False,
        redraw_random_features: bool = False,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of dimension $\mathbb{R}^{B\times Q\times D_{Q,K}}$
            ks: Keys of dimension $\mathbb{R}^{B\times K\times D_{Q,K}}$
            vs: Values of dimension $\mathbb{R}^{B\times K\tiems D_V}$
            valid_lens: Mask consisting of valid length per sequence of dimension
                $\mathbb{R}^B$ or $\mathbb{R}^{B\times K}$
            training: Boolean indicating whether currently training.
            redraw_random_features: Redraw random features used for kernel
                approximation of attention.

        Returns:
            `ctx` and `attn`, the updated values and None, respectively,
            since the attention matrix is never materialized in FAVOR+.
        """
        (B, Q, D_QK), K, D_V, H = qs.shape, ks.shape[1], vs.shape[-1], self.num_heads
        D_QK_H, D_V_H = D_QK // H, D_V // H
        qs, ks, vs = nn.Dense(D_QK)(qs), nn.Dense(D_QK)(ks), nn.Dense(D_V)(vs)
        # [B, {Q,K}, D_{QK,V}] -> [B * H, {Q,K}, D_{QK,V}_H]
        qs = qs.reshape(B, Q, H, D_QK_H).transpose(0, 2, 1, 3).reshape(-1, Q, D_QK_H)
        ks = ks.reshape(B, K, H, D_QK_H).transpose(0, 2, 1, 3).reshape(-1, K, D_QK_H)
        vs = vs.reshape(B, K, H, D_V_H).transpose(0, 2, 1, 3).reshape(-1, K, D_V_H)
        if valid_lens is not None:
            valid_lens = jnp.repeat(valid_lens, H, axis=0)
        ctx, attn = FastSoftmaxAttention(self.p_dropout)(
            qs, ks, vs, valid_lens, training, redraw_random_features
        )
        ctx = ctx.reshape(B, H, Q, D_V_H).transpose(0, 2, 1, 3).reshape(B, Q, D_V)
        return nn.Dense(D_V)(ctx), attn
