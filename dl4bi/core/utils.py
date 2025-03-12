import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
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


def nan_pad(v: jax.Array, axis: int, L: int):
    pad = [(0, 0)] * v.ndim
    L_v = v.shape[axis]
    pad[axis] = (0, L - L_v)
    return jnp.pad(v, pad, mode="constant", constant_values=jnp.nan)


def breakpoint_if_nonfinite(x):
    """Create a breakpoint when non-finite values in `x`."""
    is_finite = jnp.isfinite(x).all()

    def true_fn(x):
        pass

    def false_fn(x):
        jax.debug.breakpoint()

    lax.cond(is_finite, true_fn, false_fn, x)
