"""Utility helpers for masking, padding, and array conversion."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax
from jax.lax import conv_general_dilated
from jax.tree_util import Partial


def mask_from_valid_lens(max_len: int, valid_lens: jax.Array):
    """Return a boolean mask using `valid_lens`."""
    return jnp.arange(max_len) < valid_lens[..., None]


def exists(*args):
    """Return ``True`` when every argument is not ``None``."""
    return all([x is not None for x in args])


@jit
def safe_stack(*arrays):
    """Concatenate the non-``None`` arrays along the last axis."""
    return jnp.concat([x for x in arrays if x is not None], axis=-1)


@jit
def to_none(x: jax.Array):
    """Return ``None`` inside JIT-compatible control flow."""
    return None


@partial(jit, static_argnames=("last_n",))
def causal_moving_average(x: jax.Array, last_n: int) -> jax.Array:
    """
    Args:
        x: An array of shape [B, L].

    Returns:
        An array of shape [B, L-last_n+1].
    """
    kernel = jnp.ones((last_n, 1, 1)) / last_n  # [last_n]
    y = conv_general_dilated(
        x[..., None],  # [B, L, C=1]
        kernel,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    return y[..., 0]  # [B, L-last_n]


@partial(jit, static_argnames=("n",))
def edge_filled_centered_moving_average(x: jax.Array, n: int) -> jax.Array:
    """
    Args:
        x: An array of shape [B, L].

    Returns:
        An array of shape [B, L].
    """
    pad = (n - 1) // 2
    x_pad = jnp.pad(x, ((0, 0), (pad, pad)), mode="edge")
    y = lax.reduce_window(
        x_pad,
        init_value=0.0,
        computation=lax.add,
        window_dimensions=(1, n),
        window_strides=(1, 1),
        padding="VALID",
    )
    return y / n


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
    """Pad ``v`` with ``NaN`` values along ``axis`` up to length ``L``."""
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


def to_native(x):
    """Convert NumPy values to native python values."""
    if isinstance(x, jax.Array):
        x = np.array(x)
    if isinstance(x, np.generic):
        return x.item()
    elif isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, dict):
        return {k: to_native(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple, set)):
        return [to_native(v) for v in x]
    return x
