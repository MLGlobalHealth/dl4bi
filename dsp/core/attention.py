from collections.abc import Callable, Sequence
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import jit, lax, random, vmap
from jax.nn import dot_product_attention

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
        (B, Q, H, D_QK_H), (_, K, H, D_V_H) = qs.shape, vs.shape
        drop = nn.Dropout(self.p_dropout, deterministic=not training)
        gen_proj = lambda rng: gaussian_orf(rng, self.num_ortho_features, D_QK_H)
        init_proj = lambda: gen_proj(self.make_rng("params"))
        proj = self.variable("projections", "random", init_proj)
        if redraw_random_features:
            proj.value = gen_proj(self.make_rng("rng_extra"))
        # [B, L, H, D_H] -> [B * H, L, D_H]
        qs = qs.transpose(0, 2, 1, 3).reshape(-1, Q, D_QK_H)
        ks = ks.transpose(0, 2, 1, 3).reshape(-1, K, D_QK_H)
        vs = vs.transpose(0, 2, 1, 3).reshape(-1, K, D_V_H)
        normalizer = 1 / jnp.pow(D_QK_H, 0.25)
        phi = self.build_phi(proj.value)
        qs_prime = phi(qs * normalizer)
        ks_prime = phi(ks * normalizer)
        # NOTE: mask after phi in case phi maps zero to non-zero values
        if valid_lens is not None:
            valid_lens = jnp.repeat(valid_lens, H, axis=0)
            ks_prime *= mask_from_valid_lens(ks.shape[1], valid_lens)
        ctx = fast_attend(qs_prime, ks_prime, vs)
        # [B * H, Q, D_V_H] -> [B, Q, H, D_V_H]
        ctx = ctx.reshape(B, H, Q, D_V_H).transpose(0, 2, 1, 3)
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
            `ctx` and `attn`, the updated values and attention weights.
        """
        (B, Q, H, D_QK_H), (_, K, H, D_V_H) = qs.shape, vs.shape
        drop = nn.Dropout(self.p_dropout, deterministic=not training)
        # [B, L, H, D_H] -> [B * H, L, D_H]
        qs = qs.transpose(0, 2, 1, 3).reshape(-1, Q, D_QK_H)
        ks = ks.transpose(0, 2, 1, 3).reshape(-1, K, D_QK_H)
        vs = vs.transpose(0, 2, 1, 3).reshape(-1, K, D_V_H)
        scores = self.scorer(qs.astype(self.dtype), ks.astype(self.dtype))
        if valid_lens is not None:
            valid_lens = jnp.repeat(valid_lens, H, axis=0)
            scores = mask_attn(scores, valid_lens)
        attn = nn.softmax(scores, axis=-1)  # [B * H, Q, K]
        ctx = drop(attn) @ vs  # [B * H, Q, D_V_H]
        # [B * H, Q, D_V_H] -> [B, Q, H, D_V_H]
        ctx = ctx.reshape(B, H, Q, D_V_H).transpose(0, 2, 1, 3)
        return ctx, attn.reshape(B, H, Q, K)


class FusedAttention(nn.Module):
    r"""Performs query-key-value attention.

    Returns:
        A `FusedAttention` module.

    .. note::
        As of 2024-08-29, this requires `jax-nightly` and an NVIDIA GPU of
        Ampere architecture or above.

    .. note::
        This version does not support attention dropout since it is a fused
        softmax attention that operates on an optimized CUDA kernel. It
        also does not return the attention matrix since it never completely
        materializes it.
    """

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
        if valid_lens is None:
            valid_lens = jnp.repeat(L, B)
        # As of 2024-08-29, the CUDA kernel requires bfloat16
        return dot_product_attention(
            jnp.bfloat16(qs),
            jnp.bfloat16(ks),
            jnp.bfloat16(vs),
            # TODO(danj): remove when PR lands https://github.com/google/jax/issues/23349
            query_seq_lengths=jnp.repeat(L, B),
            key_value_seq_lengths=valid_lens,
            implementation="cudnn",
        ), None


class MultiheadAttention(nn.Module):
    r"""Performs multihead query-key-value attention.

    Args:
        attn: An attention module.
        num_heads: Number of heads for attention module.
        proj_qs: A module for projecting queries.
        proj_ks: A module for projecting keys.
        proj_vs: A module for projecting values.
        proj_out: A module for projecting output.

    Returns:
        A `MultiheadAttention` module.

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
        qs: jax.Array,
        ks: jax.Array,
        vs: jax.Array,
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


@jit
def outer_subtract(x: jax.Array, y: jax.Array):
    """Outer subtracts two arrays: `[L, D] x [K, D] -> [L, K, D]`"""
    return vmap(vmap(jnp.subtract, (None, 0)), (0, None))(x, y)


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
