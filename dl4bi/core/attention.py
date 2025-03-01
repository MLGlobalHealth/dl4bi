import warnings
from collections.abc import Callable
from functools import partial
from typing import Optional

import flax.linen as nn
import flax.linen.initializers as init
import jax
import jax.numpy as jnp
import jraph
from einops import rearrange
from jax import jit, lax, random
from jax.lax import scan

from .bias import scanned_rbf_network_bias, scanned_tisa_bias
from .mlp import MLP


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
        qs: jax.Array,  # [B, H, Q, D_qk]
        ks: jax.Array,  # [B, H, K, D_qk]
        vs: jax.Array,  # [B, H, K, D_v]
        mask: Optional[jax.Array] = None,  # [B, K]
        training: bool = False,
        redraw_random_features: bool = False,
        **kwargs,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of shape [B, H, Q, D_qk].
            ks: Keys of shape [B, H, K, D_qk].
            vs: Values of shape [B, H, K, D_v].
            mask: Mask for keys and values of shape [B, K].
            training: Boolean indicating whether currently training.
            redraw_random_features: Redraw random features used for
                softmax kernel approximation.

        Returns:
            `ctx` and `attn`, the updated values and None, respectively,
            since the attention matrix is never materialized in FAVOR+.
        """
        B, H, Q, D_qk = qs.shape
        drop = nn.Dropout(self.p_dropout, deterministic=not training)
        gen_qk_proj = lambda rng: gaussian_orf(rng, self.num_ortho_features, D_qk)
        qk_proj = self.variable(
            "projections",
            "qk_orf",
            lambda: gen_qk_proj(self.make_rng("params")),
        )
        if kwargs.get("bias") is not None:
            warnings.warn("FastAttention does not currently support bias!")
        if redraw_random_features:
            qk_proj.value = gen_qk_proj(self.make_rng("rng_extra"))
        qs, ks, vs = map(lambda x: rearrange(x, "B H L D -> (B H) L D"), (qs, ks, vs))
        normalizer = 1 / jnp.pow(D_qk, 0.25)
        phi = self.build_phi(qk_proj.value)
        qs_prime = phi(qs * normalizer)
        ks_prime = phi(ks * normalizer)
        # NOTE: mask after phi in case phi maps zero to non-zero values
        if mask is not None:  # ks_prime: [B*H, K, D]
            ks_prime *= jnp.repeat(mask, H, axis=0)[..., None]
        ctx = fast_attend(qs_prime, ks_prime, vs)
        ctx = rearrange(ctx, "(B H) Q D -> B H Q D", H=H)
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
        qs: jax.Array,  # [B, H, Q, D_qk]
        ks: jax.Array,  # [B, H, K, D_qk]
        vs: jax.Array,  # [B, H, K, D_v]
        mask: Optional[jax.Array] = None,  # [B, K]
        training: bool = False,
        **kwargs,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of shape [B, H, Q, D_qk].
            ks: Keys of shape [B, H, K, D_qk].
            vs: Values of shape [B, H, K, D_v].
            mask: Mask for keys and values of shape [B, K].
            training: Boolean indicating whether currently training.

        Returns:
            `ctx` and `attn`, the updated values and None, respectively,
            since scanned attention never materializes the attention matrix.
        """
        if kwargs.get("bias") is not None:
            warnings.warn("ScanAttention does not currently support bias!")
        return scan_attention(
            qs,
            ks,
            vs,
            mask,
            self.qs_chunk_size,
            self.ks_chunk_size,
        ), None


@partial(jit, static_argnames=("qs_chunk_size", "ks_chunk_size"))
def scan_attention(
    qs: jax.Array,
    ks: jax.Array,
    vs: jax.Array,
    mask: Optional[jax.Array] = None,
    qs_chunk_size: int = 1024,
    ks_chunk_size: int = 1024,
):
    """Scan attention based on [Flash Attention 2](https://arxiv.org/abs/2307.08691).

    Implementation inspired by [flash-attention-jax](https://github.com/lucidrains/flash-attention-jax)
    and Google's [memory-efficient-attention](https://bit.ly/4eFA4mC).
    """
    (B, H, Q, D), Q_c = qs.shape, qs_chunk_size
    Q_c = min(Q, qs_chunk_size)

    # JAX/numpy store data in row major format, so (theoretically) putting the
    # scanned axes first improves cache locality
    qs, ks, vs = map(lambda x: rearrange(x, "B H L D -> L B H D"), (qs, ks, vs))
    if mask is not None:
        mask = rearrange(mask, "B K -> K B")

    @jit
    def qs_scanner(i, _):
        qs_chunk = lax.dynamic_slice(qs, (i, 0, 0, 0), (Q_c, B, H, D))
        return i + Q_c, scan_ks(qs_chunk, ks, vs, mask, ks_chunk_size)

    i, os = scan(
        qs_scanner,
        init=0,
        xs=None,
        length=Q // Q_c,
    )
    os = rearrange(os, "C Q B H D -> B H (C Q) D")

    remainder = Q % Q_c
    if remainder:
        qs_chunk = lax.dynamic_slice(qs, (i, 0, 0, 0), (remainder, B, H, D))
        os_chunk = scan_ks(qs_chunk, ks, vs, mask, ks_chunk_size)
        os_chunk = rearrange(os_chunk, "Q B H D -> B H Q D")
        os = jnp.concatenate([os, os_chunk], axis=2)

    return os


def scan_ks(
    qs_chunk: jax.Array,
    ks: jax.Array,
    vs: jax.Array,
    mask: Optional[jax.Array] = None,
    ks_chunk_size: int = 1024,
):
    (Q_c, B, H, D), K = qs_chunk.shape, ks.shape[0]
    D_V = vs.shape[-1]
    K_c = min(K, ks_chunk_size)
    qs_chunk /= jnp.sqrt(D)

    @jit
    def ks_scanner(carry: tuple, _):
        i, os, row_maxs, row_sums = carry
        ks_chunk, vs_chunk, mask_chunk = chunk_ks(i, K_c)
        _carry = update(
            qs_chunk,
            ks_chunk,
            vs_chunk,
            mask_chunk,
            os,
            row_maxs,
            row_sums,
        )
        return (i + K_c, *_carry), None

    def chunk_ks(i, k_c):
        ks_chunk = lax.dynamic_slice(ks, (i, 0, 0, 0), (k_c, B, H, D))
        vs_chunk = lax.dynamic_slice(vs, (i, 0, 0, 0), (k_c, B, H, D_V))
        mask_chunk = jnp.array(True)
        if mask is not None:
            mask_chunk = lax.dynamic_slice(mask, (i, 0), (k_c, B))
            mask_chunk = rearrange(mask_chunk, "K B -> 1 B 1 K")
        return ks_chunk, vs_chunk, mask_chunk

    @jit
    @partial(jax.remat, prevent_cse=False)
    def update(qs_chunk, ks_chunk, vs_chunk, mask_chunk, os, row_maxs, row_sums):
        scores = jnp.einsum("Q B H D, K B H D -> Q B H K", qs_chunk, ks_chunk)
        scores = jnp.where(mask_chunk, scores, -float("inf"))
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

    os = jnp.zeros((Q_c, B, H, D_V))
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
        ks_chunk, vs_chunk, mask_chunk = chunk_ks(i, remainder)
        os, row_maxs, row_sums = update(
            qs_chunk,
            ks_chunk,
            vs_chunk,
            mask_chunk,
            os,
            row_maxs,
            row_sums,
        )

    return os / row_sums


class RBFNetworkBiasedScanAttention(nn.Module):
    r"""Performs query-key-value attention with a scan and an RBF network bias
        for reduced memory usage.

    Args:
        qs_chunk_size: Number of queries to process in each chunk of scan.
        ks_chunk_size: Number of keys to process in each chunk of scan.
        num_basis: Number of basis functions for TISA bias.

    Returns:
        An `RBFNetworkBiasedScanAttention` module.
    """

    qs_chunk_size: int = 1024
    ks_chunk_size: int = 1024
    num_basis: int = 5

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, H, Q, D_qk]
        ks: jax.Array,  # [B, H, K, D_qk]
        vs: jax.Array,  # [B, H, K, D_v]
        mask: Optional[jax.Array] = None,  # [B, K]
        training: bool = False,
        **kwargs,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of shape [B, H, Q, D_qk].
            ks: Keys of shape [B, H, Q, D_qk].
            vs: Values of shape [B, H, K, D_v].
            mask: Mask for keys and values of shape [B, K].
            training: Boolean indicating whether currently training.
            qs_locs: Query locations of shape [B, Q, D_s].
            ks_locs: Key locations of shape [B, K, D_s].

        Returns:
            `ctx` and `attn`, the updated values and None, respectively,
            since scanned attention never materializes the attention matrix.
        """
        H, F = qs.shape[1], self.num_basis
        a = self.param("a", init.constant(1), (H, F))
        b = self.param("b", init.constant(1), (H, F))
        return biased_scan_attention(
            qs,
            ks,
            vs,
            kwargs["qs_s"],  # [B, Q, D_s]
            kwargs["ks_s"],  # [B, Q, D_s]
            mask,
            self.qs_chunk_size,
            self.ks_chunk_size,
            bias_func=scanned_rbf_network_bias,
            bias_kwargs={"a": a, "b": b},
        ), None


class TISABiasedScanAttention(nn.Module):
    r"""Performs query-key-value attention with a scan and TISA bias for reduced
        memory usage.

    Args:
        qs_chunk_size: Number of queries to process in each chunk of scan.
        ks_chunk_size: Number of keys to process in each chunk of scan.
        num_basis: Number of basis functions for TISA bias.

    Returns:
        A `TISABiasedScanAttention` module.
    """

    qs_chunk_size: int = 1024
    ks_chunk_size: int = 1024
    num_basis: int = 5

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, H, Q, D_qk]
        ks: jax.Array,  # [B, H, K, D_qk]
        vs: jax.Array,  # [B, H, K, D_v]
        mask: Optional[jax.Array] = None,  # [B, K]
        training: bool = False,
        **kwargs,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of shape [B, H, Q, D_qk].
            ks: Keys of shape [B, H, K D_qk].
            vs: Values of shape [B, H, K, D_v].
            mask: Mask for keys and values of shape [B, K].
            training: Boolean indicating whether currently training.
            qs_locs: Query locations of shape [B, Q, D_s].
            ks_locs: Key locations of shape [B, K, D_s].

        Returns:
            `ctx` and `attn`, the updated values and None, respectively,
            since scanned attention never materializes the attention matrix.
        """
        H, F = qs.shape[1], self.num_basis
        a = self.param("a", init.constant(1), (H, F))
        b = self.param("b", init.constant(1), (H, F))
        c = self.param("c", init.constant(0), (H, F))
        return biased_scan_attention(
            qs,
            ks,
            vs,
            kwargs["qs_s"],  # [B, Q, D_s]
            kwargs["ks_s"],  # [B, Q, D_s]
            mask,
            self.qs_chunk_size,
            self.ks_chunk_size,
            bias_func=scanned_tisa_bias,
            bias_kwargs={"a": a, "b": b, "c": c},
        ), None


@partial(jit, static_argnames=("qs_chunk_size", "ks_chunk_size", "bias_func"))
def biased_scan_attention(
    qs: jax.Array,  # [B, H, Q, D_qk]
    ks: jax.Array,  # [B, H, K, D_qk]
    vs: jax.Array,  # [B, H, K, D_v]
    qs_meta: jax.Array,  # [B, Q, M]
    ks_meta: jax.Array,  # [B, K, M]
    mask: Optional[jax.Array] = None,  # [B, K]
    qs_chunk_size: int = 1024,
    ks_chunk_size: int = 1024,
    bias_func: Callable = scanned_tisa_bias,  # (q_meta, k_meta, **bias_kwargs) -> bias
    bias_kwargs: dict = {},
):
    (B, H, Q, D), M = qs.shape, qs_meta.shape[-1]
    Q_c = min(Q, qs_chunk_size)

    # JAX/numpy store data in row major format, so (theoretically) putting the
    # scanned axes first improves cache locality
    qs, ks, vs = map(lambda x: rearrange(x, "B H L D -> L B H D"), (qs, ks, vs))
    qs_meta, ks_meta = map(lambda x: rearrange(x, "B L M -> L B M"), (qs_meta, ks_meta))
    if mask is not None:
        mask = rearrange(mask, "B K -> K B")

    @jit
    def qs_scanner(i, _):
        qs_chunk = lax.dynamic_slice(qs, (i, 0, 0, 0), (Q_c, B, H, D))
        qs_meta_chunk = lax.dynamic_slice(qs_meta, (i, 0, 0), (Q_c, B, M))
        return i + Q_c, biased_scan_ks(
            qs_chunk,
            ks,
            vs,
            qs_meta_chunk,
            ks_meta,
            mask,
            ks_chunk_size,
            bias_func,
            bias_kwargs,
        )

    i, os = scan(
        qs_scanner,
        init=0,
        xs=None,
        length=Q // Q_c,
    )

    os = rearrange(os, "C Q B H D -> B H (C Q) D")

    remainder = Q % Q_c
    if remainder:
        qs_chunk = lax.dynamic_slice(qs, (i, 0, 0, 0), (remainder, B, H, D))
        qs_meta_chunk = lax.dynamic_slice(qs_meta, (i, 0, 0), (remainder, B, M))
        os_chunk = biased_scan_ks(
            qs_chunk,
            ks,
            vs,
            qs_meta_chunk,
            ks_meta,
            mask,
            ks_chunk_size,
            bias_func,
            bias_kwargs,
        )
        os_chunk = rearrange(os_chunk, "Q B H D -> B H Q D")
        os = jnp.concatenate([os, os_chunk], axis=2)

    return os


def biased_scan_ks(
    qs_chunk: jax.Array,  # [Q_c, B, H, D]
    ks: jax.Array,  # [K, B, H, D]
    vs: jax.Array,  # [K, B, H, D]
    qs_meta_chunk: jax.Array,  # [Q_c, B, M]
    ks_meta: jax.Array,  # [K, B, M]
    mask: Optional[jax.Array] = None,  # [K, B]
    ks_chunk_size: int = 1024,
    bias_func: Callable = scanned_tisa_bias,  # (q_meta, k_meta, **bias_kwargs) -> bias
    bias_kwargs: dict = {},
):
    (Q_c, B, H, D), (K, _, M) = qs_chunk.shape, ks_meta.shape
    K_c = min(K, ks_chunk_size)
    D_V = vs.shape[-1]
    qs_chunk /= jnp.sqrt(D)

    @jit
    def ks_scanner(carry: tuple, _):
        i, os, row_maxs, row_sums = carry
        ks_chunk, vs_chunk, ks_meta_chunk, mask_chunk = chunk_ks(i, K_c)
        _carry = update(
            qs_chunk,
            ks_chunk,
            vs_chunk,
            qs_meta_chunk,
            ks_meta_chunk,
            mask_chunk,
            os,
            row_maxs,
            row_sums,
            bias_kwargs,
        )
        return (i + K_c, *_carry), None

    def chunk_ks(i, k_c):
        ks_chunk = lax.dynamic_slice(ks, (i, 0, 0, 0), (k_c, B, H, D))
        vs_chunk = lax.dynamic_slice(vs, (i, 0, 0, 0), (k_c, B, H, D_V))
        ks_meta_chunk = lax.dynamic_slice(ks_meta, (i, 0, 0), (k_c, B, M))
        mask_chunk = jnp.array(True)
        if mask is not None:
            mask_chunk = lax.dynamic_slice(mask, (i, 0), (k_c, B))
            mask_chunk = rearrange(mask_chunk, "K B -> 1 B 1 K")
        return ks_chunk, vs_chunk, ks_meta_chunk, mask_chunk

    @jit
    @partial(jax.remat, prevent_cse=False)
    def update(
        qs_chunk,
        ks_chunk,
        vs_chunk,
        qs_meta_chunk,
        ks_meta_chunk,
        mask_chunk,
        os,
        row_maxs,
        row_sums,
        bias_kwargs,
    ):
        qs_meta_chunk_ = rearrange(qs_meta_chunk, "Q B M -> B Q M")
        ks_meta_chunk_ = rearrange(ks_meta_chunk, "K B M -> B K M")
        bias = bias_func(qs_meta_chunk_, ks_meta_chunk_, **bias_kwargs)
        bias = rearrange(bias, "B H Q K -> Q B H K")
        scores = jnp.einsum("Q B H D, K B H D -> Q B H K", qs_chunk, ks_chunk) + bias
        scores = jnp.where(mask_chunk, scores, -float("inf"))
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

    os = jnp.zeros((Q_c, B, H, D_V))
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
        ks_chunk, vs_chunk, ks_meta_chunk, mask_chunk = chunk_ks(i, remainder)
        os, row_maxs, row_sums = update(
            qs_chunk,
            ks_chunk,
            vs_chunk,
            qs_meta_chunk,
            ks_meta_chunk,
            mask_chunk,
            os,
            row_maxs,
            row_sums,
            bias_kwargs,
        )

    return os / row_sums


class Attention(nn.Module):
    r"""Performs standard Attention.

    Args:
        p_dropout: A dropout rate.
        dtype: A data type to use for calculations.

    Returns:
        An `Attention` module.
    """

    p_dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, H, Q, D_qk]
        ks: jax.Array,  # [B, H, K, D_qk]
        vs: jax.Array,  # [B, H, K, D_v]
        mask: Optional[jax.Array] = None,  # [B, K]
        training: bool = False,
        **kwargs,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of shape [B, H, Q, D_q].
            ks: Keys of shape [B, H, K, D_k].
            vs: Values of shape [B, H, K, D_v].
            mask: Mask for keys and values of shape [B, K].
            training: Boolean indicating whether currently training.

        Returns:
            `ctx` and `attn`, the updated values and attention weights.
        """
        D_qk = qs.shape[-1]
        drop = nn.Dropout(self.p_dropout, deterministic=not training)
        scores = jnp.einsum("B H Q D, B H K D -> B H Q K", qs, ks) / jnp.sqrt(D_qk)
        scores += kwargs.get("bias", 0)
        if mask is not None:
            scores = jnp.where(mask[:, None, None, :], scores, -jnp.inf)
        attn = nn.softmax(scores, axis=-1)  # [B, H, Q, K]
        ctx = jnp.einsum("B H Q K, B H K D -> B H Q D", drop(attn), vs)
        return ctx, attn


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
        qs: jax.Array,  # [B, Q, D_q]
        ks: jax.Array,  # [B, K, D_k]
        vs: jax.Array,  # [B, K, D_v]
        mask: Optional[jax.Array] = None,
        training: bool = False,
        **kwargs,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of shape [B, Q, D_q].
            ks: Keys of shape [B, K, D_k].
            vs: Values of shape [B, K, D_v].
            mask: Mask for keys and values of shape [B, K].
            training: Boolean indicating whether currently training.
            kwargs: Additional kwargs passed on to attention module.

        Returns:
            `ctx` and `attn`, the updated values and attention weights.
        """
        H = self.num_heads
        qs, ks, vs = self.proj_qs(qs), self.proj_ks(ks), self.proj_vs(vs)
        qs, ks, vs = map(
            lambda x: rearrange(x, "B L (H D) -> B H L D", H=H), (qs, ks, vs)
        )
        ctx, attn = self.attn(qs, ks, vs, mask, training, **kwargs)
        ctx = rearrange(ctx, "B H L D -> B L (H D)")
        return self.proj_out(ctx), attn


class MultiHeadGraphAttention(nn.Module):
    r"""Performs multihead query-key-value attention.

    Args:
        num_heads: Number of heads for attention module.
        proj_qs: A module for projecting queries.
        proj_ks: A module for projecting keys.
        proj_vs: A module for projecting values.
        proj_out: A module for projecting output.

    Returns:
        A `MultiHeadGraphAttention` module.

    """

    num_heads: int = 4
    proj_qks: nn.Module = MLP([64])
    proj_vs: nn.Module = MLP([64])
    proj_out: nn.Module = MLP([64])

    @nn.compact
    def __call__(self, g: jraph.GraphsTuple, training: bool = False, **kwargs):
        r"""Performs forward pass of network.

        Args:
            g: A `jraph.GraphsTuple`.
            training: Boolean indicating whether currently training.
            kwargs: Additional kwargs passed on to attention module.

        Returns:
            `ctx` and `attn`, the updated values and attention weights.
        """
        nodes, _edges, receivers, senders, globals, _n_node, _n_edge = g
        N, H = nodes.shape[0], self.num_heads
        to_mh = jit(lambda n: rearrange(n, "N (H D) -> N H D", H=H))
        qks, vs = self.proj_qks(nodes), self.proj_vs(nodes)
        qks, vs = map(to_mh, (qks, vs))
        qks_r, qks_s = qks[receivers], qks[senders]
        scores = jnp.einsum("N H D, N H D -> N H", qks_r, qks_s)
        scores += kwargs.get("bias", 0)
        bucket_size = kwargs.get("bucket_size")
        attn = _graph_segment_softmax(scores, receivers, N, bucket_size)
        ctx = _graph_conv(senders, receivers, vs, attn, N, bucket_size)
        return self.proj_out(ctx), attn


@partial(jit, static_argnames=("num_segments", "bucket_size"))
def _graph_segment_softmax(
    logits: jax.Array,
    segment_ids: jax.Array,
    num_segments: int,
    bucket_size: Optional[int] = None,
):
    """Segment softmax with `bucket_size` control for numerical stability.

    Based on [jraph's implementation](https://github.com/google-deepmind/jraph/blob/51f5990104f7374492f8f3ea1cbc47feb411c69c/jraph/_src/utils.py#L343).
    """
    maxs = jax.ops.segment_max(
        logits,
        segment_ids,
        num_segments,
        indices_are_sorted=True,
    )
    logits = jnp.exp(logits - maxs[segment_ids])
    normalizers = jax.ops.segment_sum(
        logits,
        segment_ids,
        num_segments,
        indices_are_sorted=True,
        bucket_size=bucket_size,
    )
    normalizers = normalizers[segment_ids]
    softmax = logits / normalizers
    return softmax


@partial(jit, static_argnames=("num_segments", "bucket_size"))
def _graph_conv(
    senders: jax.Array,
    receivers: jax.Array,
    nodes: jax.Array,
    attn: jax.Array,
    num_segments: int,
    bucket_size: Optional[int],
):
    from_mh = lambda n: rearrange(n, "N H D -> N (H D)")
    messages = from_mh(attn[..., None] * nodes[senders])
    return jax.ops.segment_sum(
        messages,
        receivers,
        num_segments,
        indices_are_sorted=True,
        bucket_size=bucket_size,
    )


class DeepKernelAttention(nn.Module):
    r"""Performs query-key-value attention with learned feature maps.

    Args:
        num_heads: Number of attention heads.
        proj_qks: A module for projecting queries.
        proj_vs: A module for projecting values.
        dtype: A data type to use for calculations.

    Returns:
        An `DeepKernelAttention` module.
    """

    num_heads: int = 1
    proj_qks: Callable = MLP([128, 128], nn.gelu)
    proj_vs: nn.Module = MLP([128, 64], nn.gelu)
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, D_qk]
        ks: jax.Array,  # [B, K, D_qk]
        vs: jax.Array,  # [B, K, D_v]
        mask: Optional[jax.Array] = None,
        training: bool = False,
        **kwargs,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of shape [B, Q, D_qk].
            ks: Keys of shape [B, K, D_qk].
            vs: Values of shape [B, K, D_v].
            mask: Mask for keys and values of shape [B, K].
            training: Boolean indicating whether currently training.
            kwargs: Additional kwargs passed on to attention module.

        Returns:
            `ctx` and `attn`, the updated values and `None` for the `attn`
            since it is never materialized.
        """
        (K, D), H = ks.shape[-2:], self.num_heads
        if kwargs.get("bias") is not None:
            warnings.warn("DeepKernelAttention does not support bias!")
        stack = lambda *args: jnp.concatenate(args, axis=-1)
        qs, ks = stack(kwargs["qs_s"], qs), stack(kwargs["ks_s"], ks)
        qs, ks = map(
            lambda x: self.proj_qks(x).astype(self.dtype) / jnp.pow(D, 0.25), (qs, ks)
        )
        vs = self.proj_vs(vs).astype(self.dtype)
        if mask is not None:
            ks *= mask[..., None]
            vs /= mask.sum(axis=-1, keepdims=True)[:, None]
        qs, ks, vs = map(
            lambda x: rearrange(x, "B L (H D) -> B H L D", H=H), (qs, ks, vs)
        )
        kvs = jnp.einsum("B H K D, B H K V -> B H D V", ks, vs)
        ctx = jnp.einsum("B H Q D, B H D V -> B H Q V", qs, kvs)
        ctx = rearrange(ctx, "B H Q V -> B Q (H V)")
        return nn.LayerNorm()(ctx), None
