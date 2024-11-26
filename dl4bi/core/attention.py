import warnings
from collections.abc import Callable, Sequence
from functools import partial
from typing import Optional

import flax.linen as nn
import flax.linen.initializers as init
import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from jax import jit, lax, random, vmap
from jax.lax import scan
from jax.nn import dot_product_attention
from sps.kernels import l2_dist, outer_subtract

from .bias import rbf_basis
from .mlp import MLP
from .utils import mask_attn, mask_from_valid_lens


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

    # TODO(danj): this only returns 1s when d=1
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


def build_exp_phi(proj: jax.Array):
    return build_generalized_kernel_phi(proj, jnp.exp)


def build_elu_phi(proj: jax.Array):
    return build_generalized_kernel_phi(proj, nn.elu)


def build_gelu_phi(proj: jax.Array):
    return build_generalized_kernel_phi(proj, nn.gelu)


def build_relu_phi(proj: jax.Array):
    return build_generalized_kernel_phi(proj, nn.relu)


def build_generalized_kernel_phi(
    proj: jax.Array,
    kernel_fn: Callable,
    eps: float = 0.0,
):
    r"""Builds the generalized kernel from equation (7) in [FAVOR+](https://arxiv.org/abs/2009.14794).

    Args:
        proj: A random projection to use for transforming input features.
        kernel_fn: A callable to used as the attention kernel.
        eps: A kernel epsilon added to the transformed data.

    Returns:
        $\phi$, a function that maps data to vectors used in kernel
            approximation.
    """

    def h(x):
        m = proj.shape[0]
        return jnp.sqrt(m)  # cancels sqrt(m) in coef; google's impl has no coef

    return build_phi(h, [kernel_fn], proj, eps)


def build_phi(h: Callable, funcs: list[Callable], proj: jax.Array, eps: float = 0.0):
    r"""Builds $\phi(\mathbf{x})$ from equation (5) of [FAVOR+](https://arxiv.org/abs/2009.14794).

    Args:
        h: A coefficient function.
        funcs: The set of mappings to pass random projections through.
        proj: The random projection for input features.
        eps: A kernel epsilon added to the transformed data.

    Returns:
        $\phi$, a function that maps data to vectors used in kernel
            approximation.
    """
    m = proj.shape[0]  # m normalizes over projection features
    return jit(
        lambda x: (
            h(x)
            / jnp.sqrt(m)
            * jnp.concatenate(
                [func(jnp.einsum("...d,md->...m", x, proj)) for func in funcs],
                axis=-1,
            )
            + eps
        )
    )


# TODO(danj): implement additive bias?
# TODO(danj): remove maximum to stablize logits
class FastAttention(nn.Module):
    r"""[FAVOR+](https://arxiv.org/abs/2009.14794) implementation, Appendix B.

    Implementation based on [google-research fast attention](https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py) and [Teddy Koker's implementation](https://github.com/teddykoker/performer).
    """

    p_dropout: float = 0.0
    build_phi: Callable = build_stable_positive_softmax_phi
    num_ortho_features: int = 64

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, H, D_QK_H]
        ks: jax.Array,  # [B, K, H D_QK_H]
        vs: jax.Array,  # [B, K, H, D_H]
        valid_lens: Optional[jax.Array] = None,  # [B]
        training: bool = False,
        redraw_random_features: bool = False,
        **kwargs,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of dimension $\mathbb{R}^{B\times Q\times H\times D_{Q,K}_H}$
            ks: Keys of dimension $\mathbb{R}^{B\times K\times H\times D_{Q,K}_H}$
            vs: Values of dimension $\mathbb{R}^{B\times K\times H\times D_V_H}$
            valid_lens: Mask consisting of valid length per sequence of dimension
                $\mathbb{R}^B$.
            training: Boolean indicating whether currently training.
            redraw_random_features: Redraw random features used for
                softmax kernel approximation.

        Returns:
            `ctx` and `attn`, the updated values and None, respectively,
            since the attention matrix is never materialized in FAVOR+.
        """
        (H, D_QK_H), K = qs.shape[-2:], ks.shape[1]
        drop = nn.Dropout(self.p_dropout, deterministic=not training)
        gen_qk_proj = lambda rng: gaussian_orf(rng, self.num_ortho_features, D_QK_H)
        qk_proj = self.variable(
            "projections",
            "qk_orf",
            lambda: gen_qk_proj(self.make_rng("params")),
        )
        if kwargs.get("bias") is not None:
            warnings.warn("FastAttention does not currently support bias!")
        if redraw_random_features:
            qk_proj.value = gen_qk_proj(self.make_rng("rng_extra"))
        qs, ks, vs = map(lambda x: rearrange(x, "B L H D -> (B H) L D"), (qs, ks, vs))
        normalizer = 1 / jnp.pow(D_QK_H, 0.25)
        phi = self.build_phi(qk_proj.value)
        qs_prime = phi(qs * normalizer)
        ks_prime = phi(ks * normalizer)
        # NOTE: mask after phi in case phi maps zero to non-zero values
        if valid_lens is not None:
            valid_lens = jnp.repeat(valid_lens, H, axis=0)
            ks_prime *= mask_from_valid_lens(K, valid_lens)
        ctx = fast_attend(qs_prime, ks_prime, vs)
        ctx = rearrange(ctx, "(B H) Q D -> B Q H D", H=H)
        return drop(ctx), None


# TODO(danj): normalization
# TODO(danj): add parameter to scale projection
class DistanceBiasedFastAttention(nn.Module):
    r"""[FAVOR+](https://arxiv.org/abs/2009.14794) implementation, Appendix B.

    Implementation based on [google-research fast attention](https://github.com/google-research/google-research/blob/master/performer/fast_attention/jax/fast_attention.py) and [Teddy Koker's implementation](https://github.com/teddykoker/performer).
    """

    p_dropout: float = 0.0
    build_phi: Callable = build_stable_positive_softmax_phi
    num_ortho_features: int = 64

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, H, D_QK_H]
        ks: jax.Array,  # [B, K, H D_QK_H]
        vs: jax.Array,  # [B, K, H, D_H]
        valid_lens: Optional[jax.Array] = None,  # [B]
        training: bool = False,
        redraw_random_features: bool = False,
        **kwargs,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of dimension $\mathbb{R}^{B\times Q\times H\times D_{Q,K}_H}$
            ks: Keys of dimension $\mathbb{R}^{B\times K\times H\times D_{Q,K}_H}$
            vs: Values of dimension $\mathbb{R}^{B\times K\times H\times D_V_H}$
            valid_lens: Mask consisting of valid length per sequence of dimension
                $\mathbb{R}^B$.
            training: Boolean indicating whether currently training.
            redraw_random_features: Redraw random features used for
                softmax kernel approximation.

        Returns:
            `ctx` and `attn`, the updated values and None, respectively,
            since the attention matrix is never materialized in FAVOR+.
        """
        qs_s, ks_s = kwargs["qs_s"], kwargs["ks_s"]
        (H, D_QK_H), K, S = qs.shape[-2:], ks.shape[1], qs_s.shape[-1]
        drop = nn.Dropout(self.p_dropout, deterministic=not training)
        # TODO(jhoots): change this to D_QK_H + D_S
        gen_proj = lambda D, rng: gaussian_orf(rng, self.num_ortho_features, D)
        # TODO(joots): change gen_s_proj to RFFS
        gen_qk_proj, gen_s_proj = partial(gen_proj, D_QK_H), partial(gen_proj, S)
        qk_proj = self.variable(
            "projections",
            "qk_orf",
            lambda: gen_qk_proj(self.make_rng("params")),
        )
        # TODO(jhoots): change to RFFs proj into 16
        s_proj = self.variable(
            "projections",
            "s_orf",
            lambda: gen_s_proj(self.make_rng("params")),
        )
        if kwargs.get("bias") is not None:
            warnings.warn("FastAttention does not currently support bias!")
        if redraw_random_features:
            qk_proj.value = gen_qk_proj(self.make_rng("rng_extra"))
            s_proj.value = gen_s_proj(self.make_rng("rng_extra"))
        qs, ks, vs = map(lambda x: rearrange(x, "B L H D -> (B H) L D"), (qs, ks, vs))
        normalizer = 1 / jnp.pow(D_QK_H, 0.25)
        phi = self.build_phi(qk_proj.value)
        # TODO(jhoots): concatenate RFF s's with qs/ks after normalization
        qs_prime = phi(qs * normalizer)
        ks_prime = phi(ks * normalizer)
        # NOTE: mask after phi in case phi maps zero to non-zero values
        if valid_lens is not None:
            valid_lens = jnp.repeat(valid_lens, H, axis=0)
            ks_prime *= mask_from_valid_lens(K, valid_lens)
        ctx = fast_attend(qs_prime, ks_prime, vs)
        ctx = rearrange(ctx, "(B H) Q D -> B Q H D", H=H)
        return drop(ctx), None


@jit
def fast_attend(
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
    d_inv = (1 / buf_4)[..., None]
    return d_inv * buf_3


class ScanAttention(nn.Module):
    r"""Performs query-key-value attention with a scan for reduced memory usage.

    .. warning::
        This does not currently support bias.

    Args:
        qs_chunk_size: Number of queries to process in each chunk of scan.
        ks_chunk_size: Number of keys to process in each chunk of scan.

    Returns:
        A `ScanAttention` module.
    """

    qs_chunk_size: int = 1024
    ks_chunk_size: int = 1024

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, H, D_QK_H]
        ks: jax.Array,  # [B, K, H D_QK_H]
        vs: jax.Array,  # [B, K, H, D_H]
        valid_lens: Optional[jax.Array] = None,  # [B]
        training: bool = False,
        **kwargs,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of dimension $\mathbb{R}^{B\times Q\times H\times D_QK_H}$
            ks: Keys of dimension $\mathbb{R}^{B\times K\times H\times D_QK_H}$
            vs: Values of dimension $\mathbb{R}^{B\times K\times H\times  D_V_H}$
            valid_lens: Mask consisting of valid length per sequence of dimension
                $\mathbb{R}^B$.
            training: Boolean indicating whether currently training.

        Returns:
            `ctx` and `attn`, the updated values and None, respectively,
            since scanned attention never materializes the attention matrix.
        """
        if kwargs.get("bias") is not None:
            warnings.warn("ScanAttention does not currently support bias!")
        ks_mask = None
        if valid_lens is not None:
            ks_mask = mask_from_valid_lens(ks.shape[1], valid_lens)[..., 0]
        return scan_attention(
            qs,
            ks,
            vs,
            ks_mask,
            self.qs_chunk_size,
            self.ks_chunk_size,
        ), None


@partial(jit, static_argnames=("qs_chunk_size", "ks_chunk_size"))
def scan_attention(
    qs: jax.Array,
    ks: jax.Array,
    vs: jax.Array,
    ks_mask: Optional[jax.Array] = None,
    qs_chunk_size: int = 1024,
    ks_chunk_size: int = 1024,
):
    """Scan attention based on [Flash Attention 2](https://arxiv.org/abs/2307.08691).

    Implementation inspired by [flash-attention-jax](https://github.com/lucidrains/flash-attention-jax)
    and Google's [memory-efficient-attention](https://bit.ly/4eFA4mC).
    """
    (B, Q, H, D), Q_c = qs.shape, qs_chunk_size
    Q_c = min(Q, qs_chunk_size)

    # JAX/numpy store data in row major format, so (theoretically) putting the
    # scanned axes first improves cache locality
    qs, ks, vs = map(lambda x: rearrange(x, "B L H D -> L B H D"), (qs, ks, vs))
    if ks_mask is not None:
        ks_mask = rearrange(ks_mask, "B K -> K B")

    @jit
    def qs_scanner(i, _):
        qs_chunk = lax.dynamic_slice(qs, (i, 0, 0, 0), (Q_c, B, H, D))
        return i + Q_c, scan_ks(qs_chunk, ks, vs, ks_mask, ks_chunk_size)

    i, os = scan(
        qs_scanner,
        init=0,
        xs=None,
        length=Q // Q_c,
    )
    os = rearrange(os, "C Q B H D -> B (C Q) H D")

    remainder = Q % Q_c
    if remainder:
        qs_chunk = lax.dynamic_slice(qs, (i, 0, 0, 0), (remainder, B, H, D))
        os_chunk = scan_ks(qs_chunk, ks, vs, ks_mask, ks_chunk_size)
        os_chunk = rearrange(os_chunk, "Q B H D -> B Q H D")
        os = jnp.concatenate([os, os_chunk], axis=1)

    return os


def scan_ks(
    qs_chunk: jax.Array,
    ks: jax.Array,
    vs: jax.Array,
    ks_mask: Optional[jax.Array] = None,
    ks_chunk_size: int = 1024,
):
    (Q_c, B, H, D), K = qs_chunk.shape, ks.shape[0]
    K_c = min(K, ks_chunk_size)
    qs_chunk /= jnp.sqrt(D)

    @jit
    def ks_scanner(carry: tuple, _):
        i, os, row_maxs, row_sums = carry
        ks_chunk, vs_chunk, ks_mask_chunk = chunk_ks(i, K_c)
        _carry = update(
            qs_chunk,
            ks_chunk,
            vs_chunk,
            ks_mask_chunk,
            os,
            row_maxs,
            row_sums,
        )
        return (i + K_c, *_carry), None

    def chunk_ks(i, k_c):
        ks_chunk = lax.dynamic_slice(ks, (i, 0, 0, 0), (k_c, B, H, D))
        vs_chunk = lax.dynamic_slice(vs, (i, 0, 0, 0), (k_c, B, H, D))
        ks_mask_chunk = jnp.array(True)
        if ks_mask is not None:
            ks_mask_chunk = lax.dynamic_slice(ks_mask, (i, 0), (k_c, B))
            ks_mask_chunk = rearrange(ks_mask_chunk, "K B -> 1 B 1 K")
        return ks_chunk, vs_chunk, ks_mask_chunk

    @jit
    @partial(jax.remat, prevent_cse=False)
    def update(qs_chunk, ks_chunk, vs_chunk, ks_mask_chunk, os, row_maxs, row_sums):
        scores = jnp.einsum("Q B H D, K B H D -> Q B H K", qs_chunk, ks_chunk)
        scores = jnp.where(ks_mask_chunk, scores, -float("inf"))
        row_maxs_chunk = jnp.max(scores, axis=-1, keepdims=True)
        new_row_maxs = jnp.maximum(row_maxs_chunk, row_maxs)
        exp_scores = jnp.exp(scores - new_row_maxs)
        row_sums_chunk = jnp.sum(exp_scores, axis=-1, keepdims=True)
        os_chunk = jnp.einsum("Q B H K, K B H D -> Q B H D", exp_scores, vs_chunk)
        exp_row_maxs_diff = jnp.exp(row_maxs - new_row_maxs)
        new_row_sums = exp_row_maxs_diff * row_sums + row_sums_chunk
        os *= exp_row_maxs_diff
        os += os_chunk
        return os, new_row_maxs, new_row_sums

    os = jnp.zeros((Q_c, B, H, D))
    row_sums = jnp.zeros((Q_c, B, H, 1))
    row_maxs = jnp.full((Q_c, B, H, 1), -float("inf"))

    (i, os, row_maxs, row_sums), _ = scan(
        ks_scanner,
        init=(0, os, row_maxs, row_sums),
        xs=None,
        length=K // K_c,
    )

    # last block
    remainder = K % K_c
    if remainder:
        ks_chunk, vs_chunk, ks_mask_chunk = chunk_ks(i, remainder)
        os, row_maxs, row_sums = update(
            qs_chunk,
            ks_chunk,
            vs_chunk,
            ks_mask_chunk,
            os,
            row_maxs,
            row_sums,
        )

    return os / row_sums


class TISABiasedScanAttention(nn.Module):
    r"""Performs query-key-value attention with a scan and TISA bias for reduced
        memory usage.

    Args:
        qs_chunk_size: Number of queries to process in each chunk of scan.
        ks_chunk_size: Number of keys to process in each chunk of scan.
        num_basis: Number of basis functions for TISA bias.

    Returns:
        A `ScanTISABiasedAttention` module.
    """

    qs_chunk_size: int = 1024
    ks_chunk_size: int = 1024
    num_basis: int = 5

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, H, D_QK_H]
        ks: jax.Array,  # [B, K, H D_QK_H]
        vs: jax.Array,  # [B, K, H, D_H]
        valid_lens: Optional[jax.Array] = None,  # [B]
        training: bool = False,
        **kwargs,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of dimension $\mathbb{R}^{B\times Q\times H\times D_QK_H}$
            ks: Keys of dimension $\mathbb{R}^{B\times K\times H\times D_QK_H}$
            vs: Values of dimension $\mathbb{R}^{B\times K\times H\times  D_V_H}$
            qs_locs: Query locations of dimension $\mathbb{R}^{B\times Q\times S}$
            ks_locs: Key locations of dimension $\mathbb{R}^{B\times K\times S}$
            valid_lens: Mask consisting of valid length per sequence of dimension
                $\mathbb{R}^B$.
            training: Boolean indicating whether currently training.

        Returns:
            `ctx` and `attn`, the updated values and None, respectively,
            since scanned attention never materializes the attention matrix.
        """
        K, H, F = ks.shape[1], qs.shape[2], self.num_basis
        a = self.param("a", init.constant(1), (H * F,))
        b = self.param("b", init.constant(1), (H * F,))
        c = self.param("c", init.constant(0), (H * F,))
        ks_mask = None
        if valid_lens is not None:
            ks_mask = mask_from_valid_lens(K, valid_lens)[..., 0]
        return scan_tisa_biased_attention(
            qs,
            ks,
            vs,
            kwargs["qs_s"],  # [B, Q, S]
            kwargs["ks_s"],  # [B, Q, S]
            a,
            b,
            c,
            ks_mask,
            self.qs_chunk_size,
            self.ks_chunk_size,
        ), None


@partial(jit, static_argnames=("qs_chunk_size", "ks_chunk_size"))
def scan_tisa_biased_attention(
    qs: jax.Array,  # [B, Q, H, D_QK_H]
    ks: jax.Array,  # [B, K, H D_QK_H]
    vs: jax.Array,  # [B, K, H, D_H]
    qs_s: jax.Array,  # [B, Q, S]
    ks_s: jax.Array,  # [B, K, S]
    a: jax.Array,  # [H, F]
    b: jax.Array,  # [H, F]
    c: jax.Array,  # [H, F]
    ks_mask: Optional[jax.Array] = None,  # [B, K]
    qs_chunk_size: int = 1024,
    ks_chunk_size: int = 1024,
):
    (B, Q, H, D), S = qs.shape, qs_s.shape[-1]
    Q_c = min(Q, qs_chunk_size)

    # JAX/numpy store data in row major format, so (theoretically) putting the
    # scanned axes first improves cache locality
    qs, ks, vs = map(lambda x: rearrange(x, "B L H D -> L B H D"), (qs, ks, vs))
    qs_s, ks_s = map(lambda x: rearrange(x, "B L S -> L B S"), (qs_s, ks_s))
    if ks_mask is not None:
        ks_mask = rearrange(ks_mask, "B K -> K B")

    @jit
    def qs_scanner(i, _):
        qs_chunk = lax.dynamic_slice(qs, (i, 0, 0, 0), (Q_c, B, H, D))
        qs_s_chunk = lax.dynamic_slice(qs_s, (i, 0, 0), (Q_c, B, S))
        return i + Q_c, tisa_biased_scan_ks(
            qs_chunk,
            ks,
            vs,
            qs_s_chunk,
            ks_s,
            a,
            b,
            c,
            ks_mask,
            ks_chunk_size,
        )

    i, os = scan(
        qs_scanner,
        init=0,
        xs=None,
        length=Q // Q_c,
    )

    os = rearrange(os, "C Q B H D -> B (C Q) H D")

    remainder = Q % Q_c
    if remainder:
        qs_chunk = lax.dynamic_slice(qs, (i, 0, 0, 0), (remainder, B, H, D))
        qs_s_chunk = lax.dynamic_slice(qs_s, (i, 0, 0), (remainder, B, S))
        os_chunk = tisa_biased_scan_ks(
            qs_chunk,
            ks,
            vs,
            qs_s_chunk,
            ks_s,
            a,
            b,
            c,
            ks_mask,
            ks_chunk_size,
        )
        os_chunk = rearrange(os_chunk, "Q B H D -> B Q H D")
        os = jnp.concatenate([os, os_chunk], axis=1)

    return os


def tisa_biased_scan_ks(
    qs_chunk: jax.Array,  # [Q_c, B, H, D]
    ks: jax.Array,  # [K, B, H, D]
    vs: jax.Array,  # [K, B, H, D]
    qs_s_chunk: jax.Array,  # [Q_c, B, S]
    ks_s: jax.Array,  # [K, B, S]
    a: jax.Array,  # [H * F]
    b: jax.Array,  # [H * F]
    c: jax.Array,  # [H * F]
    ks_mask: Optional[jax.Array] = None,  # [K, B]
    ks_chunk_size: int = 1024,
):
    (Q_c, B, H, D), (K, _, S) = qs_chunk.shape, ks_s.shape
    K_c, F = min(K, ks_chunk_size), a.size // H
    qs_chunk /= jnp.sqrt(D)

    @jit
    def ks_scanner(carry: tuple, _):
        i, os, row_maxs, row_sums = carry
        ks_chunk, vs_chunk, ks_s_chunk, ks_mask_chunk = chunk_ks(i, K_c)
        _carry = update(
            qs_chunk,
            ks_chunk,
            vs_chunk,
            qs_s_chunk,
            ks_s_chunk,
            ks_mask_chunk,
            a,
            b,
            c,
            os,
            row_maxs,
            row_sums,
        )
        return (i + K_c, *_carry), None

    def chunk_ks(i, k_c):
        ks_chunk = lax.dynamic_slice(ks, (i, 0, 0, 0), (k_c, B, H, D))
        vs_chunk = lax.dynamic_slice(vs, (i, 0, 0, 0), (k_c, B, H, D))
        ks_s_chunk = lax.dynamic_slice(ks_s, (i, 0, 0), (k_c, B, S))
        ks_mask_chunk = jnp.array(True)
        if ks_mask is not None:
            ks_mask_chunk = lax.dynamic_slice(ks_mask, (i, 0), (k_c, B))
            ks_mask_chunk = rearrange(ks_mask_chunk, "K B -> 1 B 1 K")
        return ks_chunk, vs_chunk, ks_s_chunk, ks_mask_chunk

    @jit
    @partial(jax.remat, prevent_cse=False)
    def update(
        qs_chunk,
        ks_chunk,
        vs_chunk,
        qs_locs_chunk,
        ks_locs_chunk,
        ks_mask_chunk,
        a,
        b,
        c,
        os,
        row_maxs,
        row_sums,
    ):
        # TODO(danj): cleanup TISA
        q_c, k_c = qs_chunk.shape[0], ks_chunk.shape[0]
        qs_locs_chunk_ = rearrange(qs_locs_chunk, "Q B S -> B Q S")
        ks_locs_chunk_ = rearrange(ks_locs_chunk, "K B S -> B K S")
        d = vmap(l2_dist)(qs_locs_chunk_, ks_locs_chunk_)
        bias = vmap(rbf_basis, in_axes=(None, 0, 0, 0), out_axes=1)(d, a, b, c)
        bias = bias.reshape(B, H, F, q_c, k_c).sum(axis=2)
        bias = rearrange(bias, "B H Q K -> Q B H K")
        scores = jnp.einsum("Q B H D, K B H D -> Q B H K", qs_chunk, ks_chunk) + bias
        scores = jnp.where(ks_mask_chunk, scores, -float("inf"))
        row_maxs_chunk = jnp.max(scores, axis=-1, keepdims=True)
        new_row_maxs = jnp.maximum(row_maxs_chunk, row_maxs)
        exp_scores = jnp.exp(scores - new_row_maxs)
        row_sums_chunk = jnp.sum(exp_scores, axis=-1, keepdims=True)
        os_chunk = jnp.einsum("Q B H K, K B H D -> Q B H D", exp_scores, vs_chunk)
        exp_row_maxs_diff = jnp.exp(row_maxs - new_row_maxs)
        new_row_sums = exp_row_maxs_diff * row_sums + row_sums_chunk
        os *= exp_row_maxs_diff
        os += os_chunk
        return os, new_row_maxs, new_row_sums

    os = jnp.zeros((Q_c, B, H, D))
    row_sums = jnp.zeros((Q_c, B, H, 1))
    row_maxs = jnp.full((Q_c, B, H, 1), -float("inf"))

    (i, os, row_maxs, row_sums), _ = scan(
        ks_scanner,
        init=(0, os, row_maxs, row_sums),
        xs=None,
        length=K // K_c,
    )

    # last block
    remainder = K % K_c
    if remainder:
        ks_chunk, vs_chunk, ks_s_chunk, ks_mask_chunk = chunk_ks(i, remainder)
        os, row_maxs, row_sums = update(
            qs_chunk,
            ks_chunk,
            vs_chunk,
            qs_s_chunk,
            ks_s_chunk,
            ks_mask_chunk,
            a,
            b,
            c,
            os,
            row_maxs,
            row_sums,
        )

    return os / row_sums


class DotScorer(nn.Module):
    r"""Performs dot product attention scoring from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    $$a(\mathbf{q},\mathbf{k}) = \frac{\mathbf{q}^\intercal\mathbf{k}}{\sqrt{d}}$$

    Returns:
        An `DotScorer` module.
    """

    @nn.compact
    def __call__(self, qs: jax.Array, ks: jax.Array):
        d = qs.shape[-1]
        return jnp.einsum("bqd,bkd->bqk", qs, ks) / jnp.sqrt(d)


class AdditiveScorer(nn.Module):
    r"""Performs additive attention scoring from ["Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/abs/1409.0473).

    $$a(\mathbf{q},\mathbf{k}) = \mathbf{w}_v^\intercal(\mathbf{W}_q\mathbf{q}+\mathbf{W}_k\mathbf{k})$$

    Args:
        num_hidden: Number of features to project keys and queries.
        dtype: A data type to use for calculations.

    Returns:
        An `MultiplicativeScorer` module.
    """

    num_hidden: int = 128
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, qs: jax.Array, ks: jax.Array):
        qs = nn.Dense(self.num_hidden, use_bias=False, dtype=self.dtype)(qs)
        ks = nn.Dense(self.num_hidden, use_bias=False, dtype=self.dtype)(ks)
        # [B, Q, 1, H] + [B, 1, K, H]
        feats = jnp.expand_dims(qs, axis=2) + jnp.expand_dims(ks, axis=1)
        feats = nn.tanh(feats)
        return nn.Dense(1, use_bias=False, dtype=self.dtype)(feats).squeeze(-1)


class MultiplicativeScorer(nn.Module):
    r"""Performs multiplicative attention scoring from ["Effective Approaches to Attention-based Neural Machine Translation"](https://arxiv.org/abs/1508.04025).

    $$a(\mathbf{q},\mathbf{k}) = \mathbf{q}^\intercal\mathbf{W}_a\mathbf{k}$$

    Args:
        dtype: A data type to use for calculations.

    Returns:
        An `MultiplicativeScorer` module.
    """

    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, qs: jax.Array, ks: jax.Array):
        qs = nn.Dense(qs.shape[-1], use_bias=False, dtype=self.dtype)(qs)
        return DotScorer()(qs, ks)


class Attention(nn.Module):
    r"""Performs query-key-value attention with dropout using the given scoring function.

    Args:
        scorer: A module used to provide similarity scores between queries and keys.
        p_dropout: A dropout rate.
        dtype: A data type to use for calculations.

    Returns:
        An `Attention` module.
    """

    scorer: nn.Module = DotScorer()
    p_dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, H, D_QK_H]
        ks: jax.Array,  # [B, K, H, D_QK_H]
        vs: jax.Array,  # [B, K, H, D_H]
        valid_lens: Optional[jax.Array] = None,  # [B]
        training: bool = False,
        **kwargs,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of dimension $\mathbb{R}^{B\times Q\times H\times D_QK_H}$
            ks: Keys of dimension $\mathbb{R}^{B\times K\times H\times D_QK_H}$
            vs: Values of dimension $\mathbb{R}^{B\times K\times H\times  D_V_H}$
            valid_lens: Mask consisting of valid length per sequence of dimension
                $\mathbb{R}^B$.
            training: Boolean indicating whether currently training.

        Returns:
            `ctx` and `attn`, the updated values and attention weights.
        """
        (B, Q, H, _), K = qs.shape, ks.shape[1]
        drop = nn.Dropout(self.p_dropout, deterministic=not training)
        qs, ks, vs = map(lambda x: rearrange(x, "B L H D -> (B H) L D"), (qs, ks, vs))
        scores = self.scorer(qs.astype(self.dtype), ks.astype(self.dtype))
        bias = kwargs.get("bias", None)
        if bias is not None:
            scores += jnp.broadcast_to(bias, (B, H, Q, K)).reshape(-1, Q, K)
        if valid_lens is not None:
            valid_lens = jnp.repeat(valid_lens, H, axis=0)
            scores = mask_attn(scores, valid_lens)
        attn = nn.softmax(scores, axis=-1)  # [B * H, Q, K]
        ctx = drop(attn) @ vs  # [B * H, Q, D_V_H]
        ctx = rearrange(ctx, "(B H) Q D -> B Q H D", H=H)
        return ctx, attn.reshape(B, H, Q, K)


class FusedAttention(nn.Module):
    r"""Performs query-key-value attention.

    Returns:
        A `FusedAttention` module.

    .. note::
        Since the CUDA kernel requires `jnp.bfloat16`, this implementation
        normalizes the queries and keys using `norm_qs` and `norm_ks` in order
        to stabilize dot product logits in the attention calculation. While this
        slightly slows computation, it often obviates the need to search for an
        appropriate `scale` argument.

    .. note::
        As of 2024-08-29, this requires `jax-nightly` and an NVIDIA GPU of
        Ampere architecture or above.

    .. note::
        This version does not support attention dropout since it is a fused
        softmax attention that operates on an optimized CUDA kernel. It
        also does not return the attention matrix since it never completely
        materializes it.
    """

    norm_qs: nn.Module = nn.LayerNorm(dtype=jnp.bfloat16)
    norm_ks: nn.Module = nn.LayerNorm(dtype=jnp.bfloat16)

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, H, D_QK_H]
        ks: jax.Array,  # [B, K, H D_QK_H]
        vs: jax.Array,  # [B, K, H, D_H]
        valid_lens: Optional[jax.Array] = None,  # [B]
        training: bool = False,
        **kwargs,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of dimension $\mathbb{R}^{B\times Q\times H\times D_QK_H}$
            ks: Keys of dimension $\mathbb{R}^{B\times K\times H\times D_QK_H}$
            vs: Values of dimension $\mathbb{R}^{B\times K\times H\times  D_V_H}$
            valid_lens: Mask consisting of valid length per sequence of dimension
                $\mathbb{R}^B$.
            training: Boolean indicating whether currently training. Unused here.

        Returns:
            `ctx` and `attn`, the updated values and attention weights, which are
            `None` for this implementation.
        """
        B, L, H, D = qs.shape
        bias = kwargs.get("bias")
        if bias is not None:
            bias = jnp.bfloat16(bias)
        if valid_lens is None:
            valid_lens = jnp.repeat(L, B)
        # As of 2024-08-29, the CUDA kernel requires bfloat16
        return dot_product_attention(
            self.norm_qs(qs),
            self.norm_ks(ks),
            jnp.bfloat16(vs),
            bias,  # None or broadcastable to [B, H, Q, K]
            # TODO(danj): remove when PR lands https://github.com/google/jax/issues/23349
            query_seq_lengths=jnp.repeat(L, B),
            key_value_seq_lengths=valid_lens,
            implementation="cudnn",
        ), None


class MultiHeadAttention(nn.Module):
    r"""Performs multihead query-key-value attention.

    Args:
        attn: An attention module.
        num_heads: Number of heads for attention module.
        proj_qs: A module for projecting queries.
        proj_ks: A module for projecting keys.
        proj_vs: A module for projecting values.
        proj_out: A module for projecting output.

    Returns:
        A `MultiHeadAttention` module.

    """

    attn: nn.Module = Attention()
    num_heads: int = 4
    proj_qs: nn.Module = MLP([64])
    proj_ks: nn.Module = MLP([64])
    proj_vs: nn.Module = MLP([64])
    proj_out: nn.Module = MLP([64])

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, H, D_QK_H]
        ks: jax.Array,  # [B, K, H, D_QK_H]
        vs: jax.Array,  # [B, K, H, D_H]
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
        **kwargs,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of dimension $\mathbb{R}^{B\times Q\times D_QK}$
            ks: Keys of dimension $\mathbb{R}^{B\times K\times D_QK}$
            vs: Values of dimension $\mathbb{R}^{B\times K\times D_V}$
            valid_lens: Mask consisting of valid length per sequence of dimension
                $\mathbb{R}^B$ or $\mathbb{R}^{B\times K}$.
            training: Boolean indicating whether currently training.
            kwargs: Additional kwargs passed on to attention module.

        Returns:
            `ctx` and `attn`, the updated values and attention weights.
        """
        qs, ks, vs = self.proj_qs(qs), self.proj_ks(ks), self.proj_vs(vs)
        (B, Q, D_QK), (K, D_V), H = qs.shape, vs.shape[-2:], self.num_heads
        qs = qs.reshape(B, Q, H, D_QK // H)
        ks = ks.reshape(B, K, H, D_QK // H)
        vs = vs.reshape(B, K, H, D_V // H)
        ctx, attn = self.attn(qs, ks, vs, valid_lens, training, **kwargs)
        return self.proj_out(ctx.reshape(B, Q, D_V)), attn


@jit
def rbf_scorer(qs: jax.Array, ks: jax.Array):  # [B, L, D]
    r"""Calculates $\exp\left(-\frac{\lVert xW_x - yW_y\rVert^2}{\sqrt{d_k}}\right)$."""
    d_sq = jnp.square(vmap(outer_subtract)(qs, ks)).sum(axis=-1)
    return jnp.exp(-d_sq / jnp.sqrt(ks.shape[-1]))


@jit
def exponential_scorer(qs: jax.Array, ks: jax.Array):  # [B, L, D]
    r"""Calculates $\exp\left(-\frac{\lVert xW_x - yW_y\rVert^2}{\sqrt{d_k}}\right)$."""
    scores = jnp.einsum("bqd,bkd->bqk", qs, ks)
    return jnp.exp(scores / jnp.sqrt(ks.shape[-1]))


# TODO(danj): add learnable variance and lengthscale?
class KernelAttention(nn.Module):
    r"""Performs query-key-value attention with the given kernel.

    .. note::
        To make a kernel symmetric, provide the same dense projection
        matrix for `proj_qs` and `proj_ks`.

    Args:
        kernel_scorer: A kernel function that operates on two `[L, D]` arrays and
            returns a score for every pair.
        proj_qs: A module for projecting queries.
        proj_ks: A module for projecting keys.
        proj_vs: A module for projecting values.
        proj_out: A module for projecting output.
        dtype: A data type to use for calculations.

    Returns:
        An `KernelAttention` module.
    """

    kernel_scorer: Callable = rbf_scorer
    proj_qs: nn.Module = MLP([64])
    proj_ks: nn.Module = MLP([64])
    proj_vs: nn.Module = MLP([64])
    proj_out: nn.Module = MLP([64])
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, D_QK]
        ks: jax.Array,  # [B, K, D_QK]
        vs: jax.Array,  # [B, K, D_V]
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
        **kwargs,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of dimension $\mathbb{R}^{B\times Q\times D_QK}$
            ks: Keys of dimension $\mathbb{R}^{B\times K\times D_QK}$
            vs: Values of dimension $\mathbb{R}^{B\times K\times D_V}$
            valid_lens: Mask consisting of valid length per sequence of dimension
                $\mathbb{R}^B$ or $\mathbb{R}^{B\times K}$.
            training: Boolean indicating whether currently training.
            kwargs: Additional kwargs passed on to attention module.

        Returns:
            `ctx` and `attn`, the updated values and attention weights.
        """
        qs, ks, vs = self.proj_qs(qs), self.proj_ks(ks), self.proj_vs(vs)
        attn = self.kernel_scorer(qs.astype(self.dtype), ks.astype(self.dtype))
        if valid_lens is not None:
            attn = mask_attn(attn, valid_lens, fill=0.0)
        attn = attn / attn.sum(axis=-1)[..., None]  # [B, Q, K]
        ctx = attn @ vs  # [B, Q, D_V]
        return self.proj_out(ctx), attn


class ProductKernelAttention(nn.Module):
    r"""Performs product-kernel query-key-value attention.

    Each kernel is responsible for projecting its own input and output.
    `proj_out` projects the concatenated output of all kernels.

    Args:
        kernel_scorers: A list of kernel scorers to multiply.
        proj_out: A module for projecting output.

    Returns:
        A `MultikernelAttention` module.
    """

    kernel_scorers: Sequence[nn.Module]
    proj_out: nn.Module = MLP([64])

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, D_QK]
        ks: jax.Array,  # [B, K, D_QK]
        vs: jax.Array,  # [B, K, D_V]
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
        **kwargs,
    ):
        (B, Q, _), K = qs.shape, ks.shape[1]
        ctx = jnp.ones((B, Q, K))
        attns = []
        for kernel_scorer in self.kernel_scorers:
            k_ctx, attn = kernel_scorer(qs, ks, vs, valid_lens, training)
            ctx *= k_ctx
            attns += [attn]  # list of [B, Q, K]
        return self.proj_out(ctx), attns


class MultiKernelAttention(nn.Module):
    r"""Performs multikernel query-key-value attention.

    Each kernel is responsible for projecting its own input and output.
    `proj_out` projects the concatenated output of all kernels.

    Args:
        kernels: A list of kernel modules to use.
        proj_out: A module for projecting output.

    Returns:
        A `MultikernelAttention` module.
    """

    kernels: Sequence[nn.Module]
    proj_out: nn.Module = MLP([64])

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, D_QK]
        ks: jax.Array,  # [B, K, D_QK]
        vs: jax.Array,  # [B, K, D_V]
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
        **kwargs,
    ):
        ctxs, attns = [], []
        for kernel in self.kernels:
            ctx, attn = kernel(qs, ks, vs, valid_lens, training)
            ctxs += [ctx]  # list of [B, Q, D_V]
            attns += [attn]  # list of [B, Q, K]
        return self.proj_out(jnp.concatenate(ctxs, axis=-1)), attns


# TODO(danj): implement memory efficient version
class SpatioTemporalMLPAttention(nn.Module):
    proj_vs: nn.Module = MLP([64], nn.gelu)
    proj_attn: nn.Module = MLP([256, 256, 1], nn.gelu)
    norm: nn.Module = nn.LayerNorm()

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, F]
        ks: jax.Array,  # [B, K, F]
        vs: jax.Array,  # [B, K, D]
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
        **kwargs,
    ):
        stack = lambda *args: jnp.concatenate(args, axis=-1)
        # standard features
        (B, Q), K = qs.shape[:-1], ks.shape[1]
        qs = repeat(qs, "B Q F -> B Q K F", K=K)
        ks = repeat(ks, "B K F -> B Q K F", Q=Q)
        m = jnp.concatenate([qs, ks], axis=-1)
        # vnode (global behavior)
        vnode = kwargs.get("vnode")
        if vnode is not None:
            vnode = repeat(vnode, "B D -> B Q K D", Q=Q, K=K)
            m = stack(m, vnode)
        # spatial features
        qs_s, ks_s = kwargs.get("qs_s"), kwargs.get("ks_s")
        if qs_s is not None and ks_s is not None:
            s_diff = qs_s[..., None, :] - ks_s[:, None, ...]
            s_dist = jnp.linalg.norm(s_diff, axis=-1, keepdims=True)
            s_dist_sq = jnp.square(s_dist)
            m = stack(m, s_diff, s_dist, s_dist_sq)
        # mask
        if valid_lens is None:
            valid_lens = jnp.repeat(K, B)
        mask = v_mask = mask_from_valid_lens(K, valid_lens)  # [B, Q, 1]
        # temporal features
        qs_t, ks_t = kwargs.get("qs_t"), kwargs.get("ks_t")
        if qs_t is not None and ks_t is not None:
            t_diff = qs_t[..., None, :] - ks_t[:, None, ...]
            t_mask = (t_diff > 0)[..., 0]  # [B, Q, K]
            mask = jnp.logical_and(mask, t_mask)
            m = stack(m, t_diff)
        # TODO(danj): project attention and gate with single network?
        attn = self.proj_attn(m)[..., 0]
        attn = jnp.where(mask, attn, 0)
        # TODO(danj): use some sort of softmax normalization,
        # even though this would prevent negative attn values?
        return self.norm(attn @ self.proj_vs(vs))  # [B, Q, D]
