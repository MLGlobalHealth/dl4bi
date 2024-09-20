from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from jax import jit, lax, random, vmap
from jax.tree_util import Partial


@jit
def mask_attn(x: jax.Array, valid_lens: jax.Array, fill=-jnp.inf):
    r"""Mask `x` with `fill` using `valid_lens`.

    Args:
        x: Values of dimension $\mathbb{R}^{B\times Q\times K}$
        valid_lens: Mask consisting of valid length per sequence
            $\mathbb{R}^{B}$ or $\mathbb{R}^{B\times Q}.

    Returns:
       `x` with filled values according to mask.
    """
    B, Q, K = x.shape
    if valid_lens.ndim == 1:
        valid_lens = jnp.repeat(valid_lens, Q)
    x = x.reshape(B * Q, K)
    m = jnp.arange(K) < valid_lens.reshape(-1, 1)
    return jnp.where(m, x, fill).reshape(B, Q, K)


def mask_from_valid_lens(max_len: int, valid_lens: jax.Array):
    """Return a boolean mask using `valid_lens`.

    .. note::
        Adds a final dimension of 1, which is often used to broadcast across the
        final tensor dimension.
    """
    return (jnp.arange(max_len) < valid_lens[..., None])[..., None]


def pad_concat(x: jax.Array, y: jax.Array):
    """Concat channels of two layers, padding smaller spatial layer with zeros.

    Args:
        x: Tensor of shape [B, L_x, C_x].
        y: Tensor of shape [B, L_y, C_y].

    Returns:
        A concatenated tensor of shape [B, max(L_x, L_y), C_x + C_y].
    """
    L_x, L_y = x.shape[1], y.shape[1]
    padding = np.abs(L_x - L_y)
    is_even = padding % 2 == 0
    half_even = (padding // 2, padding // 2)
    half_odd = ((padding - 1) // 2, (padding + 1) // 2)
    p_e = Partial(jnp.pad, pad_width=((0, 0), half_even, (0, 0)), mode="reflect")
    p_o = Partial(jnp.pad, pad_width=((0, 0), half_odd, (0, 0)), mode="reflect")
    if L_x > L_y:
        y = p_e(y) if is_even else p_o(y)
    elif L_y > L_x:
        x = p_e(x) if is_even else p_o(x)
    return jnp.concatenate([x, y], axis=-1)


@partial(jit, static_argnames=("num_samples"))
def bootstrap(
    rng: jax.Array,
    x: jax.Array,  # [B, L, D]
    valid_lens: jax.Array,  # [B]
    num_samples: int = 1,
):
    """Bootstrap selects the first `valid_lens` values of `x` `num_samples` times.

    Args:
        rng: A PRNGKey.
        x: Array to bootstrap.
        valid_lens: The valid entries for every sequence in x.

    Returns:
        A bootstrap sampled array of shape [B * num_samples, L, D].
    """
    (B, L, _), K = x.shape, num_samples
    x = jnp.repeat(x, K, axis=0)
    valid_lens = jnp.repeat(valid_lens, K, axis=0)
    mask = mask_from_valid_lens(L, valid_lens).squeeze()
    rnd_idx = random.randint(rng, (B * K, L), 0, valid_lens[:, None])
    ord_idx = jnp.repeat(jnp.arange(L)[None, :], B * K, axis=0)
    boot_idx = mask * rnd_idx + ~mask * ord_idx
    return vmap(lambda row, idx: row[idx], (0, 0))(x, boot_idx), valid_lens


def breakpoint_if_nonfinite(x):
    """Create a breakpoint when non-finite values in `x`."""
    is_finite = jnp.isfinite(x).all()

    def true_fn(x):
        pass

    def false_fn(x):
        jax.debug.breakpoint()

    lax.cond(is_finite, true_fn, false_fn, x)
