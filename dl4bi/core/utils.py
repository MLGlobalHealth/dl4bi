from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, random, vmap
from jax.tree_util import Partial


def mask_from_valid_lens(max_len: int, valid_lens: jax.Array):
    """Return a boolean mask using `valid_lens`."""
    return jnp.arange(max_len) < valid_lens[..., None]


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


# TODO(danj): update this to be compatible with mask_ctx
@partial(jit, static_argnames=("num_samples"))
def bootstrap(
    rng: jax.Array,
    x: jax.Array,  # [B, L, D]
    mask: jax.Array,  # [B, L]
    num_samples: int = 1,
):
    """Bootstrap selects the first `valid_lens` values of `x` `num_samples` times.

    Args:
        rng: A PRNGKey.
        x: Array to bootstrap.
        mask: The valid entries for every sequence in x.

    Returns:
        A bootstrap sampled array of shape [B * num_samples, L, D].
    """
    (B, L, _), K = x.shape, num_samples
    x = jnp.repeat(x, K, axis=0)
    boot_mask = jnp.repeat(mask, K, axis=0)
    # TODO(danj): implement here
    rnd_idx = random.randint(rng, (B * K, L), 0, valid_lens[:, None])
    ord_idx = jnp.repeat(jnp.arange(L)[None, :], B * K, axis=0)
    boot_idx = mask * rnd_idx + ~mask * ord_idx
    return vmap(lambda row, idx: row[idx], (0, 0))(x, boot_idx), mask_boot


def breakpoint_if_nonfinite(x):
    """Create a breakpoint when non-finite values in `x`."""
    is_finite = jnp.isfinite(x).all()

    def true_fn(x):
        pass

    def false_fn(x):
        jax.debug.breakpoint()

    lax.cond(is_finite, true_fn, false_fn, x)
