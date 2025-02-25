from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import jit, random, vmap


@dataclass(frozen=True)
class BatchElement(Mapping):
    """A `BatchElement` represents a single element of a `Batch`.

    In other words, it is the same as a `Batch` object where each contained
    array no longer has the leading batch dim.
    """

    x_ctx: Optional[jax.Array] = None  # [L_ctx, D_x]
    s_ctx: Optional[jax.Array] = None  # [L_ctx, D_s]
    t_ctx: Optional[jax.Array] = None  # [L_ctx, 1]
    f_ctx: Optional[jax.Array] = None  # [L_ctx, D_f]
    valid_lens_ctx: Optional[jax.Array] = None  # [1]
    x_test: Optional[jax.Array] = None  # [L_test, D_x]
    s_test: Optional[jax.Array] = None  # [L_test, D_s]
    t_test: Optional[jax.Array] = None  # [L_test, 1]
    f_test: Optional[jax.Array] = None  # [L_test, D_f]
    valid_lens_test: Optional[jax.Array] = None  # [1]
    inv_permute_idx: Optional[jax.Array] = None  # [L]

    def update(self, **kwargs):
        """Returns a new batch with updated attributes."""
        return replace(self, **kwargs)

    def __getitem__(self, key):
        return asdict(self)[key]

    def __iter__(self):
        """Allows you to use **batch to expand as kwargs."""
        return iter(asdict(self))

    def __len__(self):
        return len(asdict(self))


# register with jax to use in compiled functions
jax.tree_util.register_pytree_node(
    BatchElement,
    lambda b: (
        (
            b.x_ctx,
            b.s_ctx,
            b.t_ctx,
            b.f_ctx,
            b.valid_lens_ctx,
            b.x_test,
            b.s_test,
            b.t_test,
            b.f_test,
            b.valid_lens_test,
            b.inv_permute_idx,
        ),
        None,
    ),
    lambda _aux, children: BatchElement(*children),
)


@dataclass(frozen=True)
class Batch(Mapping):
    """A generic `Batch` object.

    This batch object can be deconstructed with `**batch`.
    """

    x_ctx: Optional[jax.Array] = None  # [B, L_ctx, D_x]
    s_ctx: Optional[jax.Array] = None  # [B, L_ctx, D_s]
    t_ctx: Optional[jax.Array] = None  # [B, L_ctx, 1]
    f_ctx: Optional[jax.Array] = None  # [B, L_ctx, D_f]
    valid_lens_ctx: Optional[jax.Array] = None  # [B]
    x_test: Optional[jax.Array] = None  # [B, L_test, D_x]
    s_test: Optional[jax.Array] = None  # [B, L_test, D_s]
    t_test: Optional[jax.Array] = None  # [B, L_test, 1]
    f_test: Optional[jax.Array] = None  # [B, L_test, D_f]
    valid_lens_test: Optional[jax.Array] = None  # [B]
    inv_permute_idx: Optional[jax.Array] = None  # [B, L]

    def update(self, **kwargs):
        """Returns a new batch with updated attributes."""
        return replace(self, **kwargs)

    def element(self, i: int):
        d = {}
        for k, v in iter(self):
            d[k] = v
            if isinstance(v, jax.Array):
                d[k] = v[i]
        return BatchElement(**d)

    def __getitem__(self, key):
        return asdict(self)[key]

    def __iter__(self):
        """Allows you to use **batch to expand as kwargs."""
        return iter(asdict(self))

    def __len__(self):
        return len(asdict(self))


# register with jax to use in compiled functions
jax.tree_util.register_pytree_node(
    Batch,
    lambda b: (
        (
            b.x_ctx,
            b.s_ctx,
            b.t_ctx,
            b.f_ctx,
            b.valid_lens_ctx,
            b.x_test,
            b.s_test,
            b.t_test,
            b.f_test,
            b.valid_lens_test,
            b.inv_permute_idx,
        ),
        None,
    ),
    lambda _aux, children: Batch(*children),
)


@dataclass(frozen=True)
class Data:
    """A simple `Data` container.

    This class is intended for simple datasets where each element of the batch
    consists of a dataset of fixed effects `x` and function outputs `f`.
    """

    x: jax.Array  # [B, L, D_x]
    f: jax.Array  # [B, L, D_f]

    def to_batch(
        self,
        rng: jax.Array,
        num_ctx_min: int,
        num_ctx_max: int,
        num_test: int,
        independent: bool = False,
        test_includes_ctx: bool = True,
        include_inv_permute_idx: bool = False,
    ):
        """Creates a `Batch` from this `Data`.

        Args:
            rng: A PRNG.
            num_ctx_min: Minimum number of context points.
            num_ctx_max: Maximum number of context points.
            num_test: Number of test points.
            independent: Whether the subset of context points for each element
                in the batch should be selected independently.
            test_includes_ctx: Whether to include context points in the test
                set.
            include_inv_permute_idx: Whether to include the `inv_permute_idx`,
                which enables easily mapping context points back to their
                original positions in the dataset. This is useful for callbacks
                and plotting functions.
        Returns:
            A `Batch`.
        """
        return _data_to_batch(
            rng,
            self.x,
            self.f,
            num_ctx_min,
            num_ctx_max,
            num_test,
            independent,
            test_includes_ctx,
            include_inv_permute_idx,
        )


# register with jax to use in compiled functions
jax.tree_util.register_pytree_node(
    Data,
    lambda d: ((d.x, d.f), None),
    lambda _aux, children: Data(*children),
)


@partial(
    jit,
    static_argnames=(
        "num_ctx_min",
        "num_ctx_max",
        "num_test",
        "independent",
        "test_includes_ctx",
        "include_inv_permute_idx",
    ),
)
def _data_to_batch(
    rng: jax.Array,
    x: jax.Array,  # [B, L, D_x]
    f: jax.Array,  # [B, L, D_f]
    num_ctx_min: int,
    num_ctx_max: int,
    num_test: int,
    independent: bool = False,
    test_includes_ctx: bool = True,
    include_inv_permute_idx: bool = False,
):
    B = x.shape[0]
    rng_valid, rng = random.split(rng)
    valid_lens_ctx = random.randint(rng_valid, (B,), num_ctx_min, num_ctx_max)
    valid_lens_test = jnp.repeat(num_test, B)
    vbatch = vmap(
        lambda rng, v: _batch(rng, v, num_ctx_max, num_test, test_includes_ctx)
    )
    rngs = random.split(rng, B) if independent else jnp.repeat(rng, B)
    x_ctx, x_test, _ = vbatch(rngs, x)
    f_ctx, f_test, inv_permute_idx = vbatch(rngs, f)
    s_ctx, s_test = None, None
    t_ctx, t_test = None, None
    return Batch(
        x_ctx,
        s_ctx,
        t_ctx,
        f_ctx,
        valid_lens_ctx,
        x_test,
        s_test,
        t_test,
        f_test,
        valid_lens_test,
        # no need to store this if its not used
        inv_permute_idx if include_inv_permute_idx else None,
    )


@partial(jit, static_argnames=("num_ctx_max", "num_test", "test_includes_ctx"))
def _batch(
    rng: jax.Array,
    v: jax.Array,  # [L, D]
    num_ctx_max: int,
    num_test: int,
    test_includes_ctx: bool = True,
):
    L = v.shape[0]
    permute_idx = random.choice(rng, L, (L,), replace=False)
    inv_permute_idx = jnp.argsort(permute_idx)
    v_permuted = v[permute_idx]
    v_ctx = v_permuted[:num_ctx_max]
    if test_includes_ctx:
        v_test = v_permuted[:num_test]
    else:
        v_test = v_permuted[num_ctx_max : num_ctx_max + num_test]
    return v_ctx, v_test, inv_permute_idx


@dataclass(frozen=True)
class SpatialData:
    """A `SpatialData` container.

    This class is intended for datasets with a spatial dimension, `s`, which may
    have optional fixed effects, `x`, and functional output, `f`, associated with
    each location.
    """

    x: Optional[jax.Array]  # [B, [S]+, D_x] or [B, 1, D_x] or None
    s: jax.Array  # [B, [S]+, D_s]
    f: jax.Array  # [B, [S]+, D_f]

    def to_batch(
        self,
        rng: jax.Array,
        num_ctx_min: int,
        num_ctx_max: int,
        num_test: int,
        independent: bool = False,
        test_includes_ctx: bool = True,
        include_inv_permute_idx: bool = False,
    ):
        """Creates a `Batch` from this `SpatialData`.

        Args:
            rng: A PRNG.
            num_ctx_min: Minimum number of context points.
            num_ctx_max: Maximum number of context points.
            num_test: Number of test points.
            independent: Whether the subset of context points for each element
                in the batch should be selected independently.
            test_includes_ctx: Whether to include context points in the test
                set.
            include_inv_permute_idx: Whether to include the `inv_permute_idx`,
                which enables easily mapping context points back to their
                original positions in the dataset. This is useful for callbacks
                and plotting functions.
        Returns:
            A `Batch`.
        """
        return _spatial_data_to_batch(
            rng,
            self.x,
            self.s,
            self.f,
            num_ctx_min,
            num_ctx_max,
            num_test,
            independent,
            test_includes_ctx,
            include_inv_permute_idx,
        )


# register with jax to use in compiled functions
jax.tree_util.register_pytree_node(
    SpatialData,
    lambda d: ((d.x, d.s, d.f), None),
    lambda _aux, children: SpatialData(*children),
)


@partial(
    jit,
    static_argnames=(
        "num_ctx_min",
        "num_ctx_max",
        "num_test",
        "independent",
        "test_includes_ctx",
        "include_inv_permute_idx",
    ),
)
def _spatial_data_to_batch(
    rng: jax.Array,
    x: Optional[jax.Array],  # [B, [S]+, D_x] or [B, 1, D_x] or None
    s: jax.Array,  # [B, [S]+]
    f: jax.Array,  # [B, [S]+, D_f]
    num_ctx_min: int,
    num_ctx_max: int,
    num_test: int,
    independent: bool = False,
    test_includes_ctx: bool = True,
    include_inv_permute_idx: bool = False,
):
    B = s.shape[0]
    has_x = x is not None
    flatten_spatial_dims = lambda v: v.reshape(B, -1, v.shape[-1])
    x = flatten_spatial_dims(x) if has_x else None
    s = flatten_spatial_dims(s)
    f = flatten_spatial_dims(f)
    rng_valid, rng = random.split(rng)
    valid_lens_ctx = random.randint(rng_valid, (B,), num_ctx_min, num_ctx_max)
    valid_lens_test = jnp.repeat(num_test, B)
    vbatch = vmap(
        lambda rng, v: _batch(rng, v, num_ctx_max, num_test, test_includes_ctx)
    )
    rngs = random.split(rng, B) if independent else jnp.repeat(rng, B)
    x_ctx, x_test, _ = vbatch(rngs, x) if has_x else (None, None, None)
    s_ctx, s_test, _ = vbatch(rngs, s)
    f_ctx, f_test, inv_permute_idx = vbatch(rngs, f)
    t_ctx, t_test = None, None
    return Batch(
        x_ctx,
        s_ctx,
        t_ctx,
        f_ctx,
        valid_lens_ctx,
        x_test,
        s_test,
        t_test,
        f_test,
        valid_lens_test,
        # no need to store this if its not used
        inv_permute_idx if include_inv_permute_idx else None,
    )


@dataclass(frozen=True)
class TemporalData:
    """A `TemporalData` container.

    This class is intended for datasets with a temporal dimension, `t`, which may
    have optional fixed effects, `x`, and functional output, `f`, associated with
    each time.
    """

    x: Optional[jax.Array]  # [B, T, D_x] or [B, 1, D_x] None
    t: jax.Array  # [B, T, 1]
    f: jax.Array  # [B, T, D_f]

    def to_batch(
        self,
        rng: jax.Array,
        num_ctx_min: int,
        num_ctx_max: int,
        num_test: int,
        independent: bool = False,
        test_includes_ctx: bool = True,
        include_inv_permute_idx: bool = False,
    ):
        """Creates a `Batch` from this `SpatialData`.

        Args:
            rng: A PRNG.
            num_ctx_min: Minimum number of context points.
            num_ctx_max: Maximum number of context points.
            num_test: Number of test points.
            independent: Whether the subset of context points for each element
                in the batch should be selected independently.
            test_includes_ctx: Whether to include context points in the test
                set.
            include_inv_permute_idx: Whether to include the `inv_permute_idx`,
                which enables easily mapping context points back to their
                original positions in the dataset. This is useful for callbacks
                and plotting functions.
        Returns:
            A `Batch`.
        """
        return _temporal_data_to_batch(
            rng,
            self.x,
            self.t,
            self.f,
            num_ctx_min,
            num_ctx_max,
            num_test,
            independent,
            test_includes_ctx,
            include_inv_permute_idx,
        )


# register with jax to use in compiled functions
jax.tree_util.register_pytree_node(
    TemporalData,
    lambda d: ((d.x, d.t, d.f), None),
    lambda _aux, children: TemporalData(*children),
)


@partial(
    jit,
    static_argnames=(
        "num_ctx_min",
        "num_ctx_max",
        "num_test",
        "independent",
        "test_includes_ctx",
        "include_inv_permute_idx",
    ),
)
def _temporal_data_to_batch(
    rng: jax.Array,
    x: Optional[jax.Array],  # [B, T, D_x] or [B, 1, D_x] None
    t: jax.Array,  # [B, T, 1]
    f: jax.Array,  # [B, T, D_f]
    num_ctx_min: int,
    num_ctx_max: int,
    num_test: int,
    independent: bool = False,
    test_includes_ctx: bool = True,
    include_inv_permute_idx: bool = False,
):
    B = t.shape[0]
    has_x = x is not None
    rng_valid, rng = random.split(rng)
    valid_lens_ctx = random.randint(rng_valid, (B,), num_ctx_min, num_ctx_max)
    valid_lens_test = jnp.repeat(num_test, B)
    vbatch = vmap(
        lambda rng, v: _batch(rng, v, num_ctx_max, num_test, test_includes_ctx)
    )
    rngs = random.split(rng, B) if independent else jnp.repeat(rng, B)
    x_ctx, x_test, _ = vbatch(rngs, x) if has_x else (None, None, None)
    t_ctx, t_test, _ = vbatch(rngs, t)
    f_ctx, f_test, inv_permute_idx = vbatch(rngs, f)
    s_ctx, s_test = None, None
    return Batch(
        x_ctx,
        s_ctx,
        t_ctx,
        f_ctx,
        valid_lens_ctx,
        x_test,
        s_test,
        t_test,
        f_test,
        valid_lens_test,
        # no need to store this if its not used
        inv_permute_idx if include_inv_permute_idx else None,
    )


class SpatiotemporalTask(Enum):
    Forecast = "forecast"
    Interpolate = "interpolate"


@dataclass(frozen=True)
class SpatiotemporalData:
    """A `SpatiotemporalData` container.

    This class is intended for datasets with a spatial dimension, `s`, a
    temporal dimension, `t`, optional fixed effects per time step per location, `x`,
    and the functional output associated with each time step and location.

    .. note::
        Unlike the other `Data` classes, the batch dimension here is the time, `T`.

    .. warning::
        This container assumes that that timesteps are ordered from least to
        most recent.
    """

    x: Optional[jax.Array]  # [T, [S]+, D_x] or [T, 1, D_x] or [1, 1, D_x] or None
    s: jax.Array  # [T, [S]+, D_s]
    t: jax.Array  # [T, [S]+, 1] or [T, 1, 1]
    f: jax.Array  # [T, [S]+, D_f]

    def to_batch(
        self,
        rng: jax.Array,
        task: SpatiotemporalTask,
        fixed_t_step: int,
        num_t_per_element: int,
        num_ctx_min_per_t: int,
        num_ctx_max_per_t: int,
        independent: bool,
        num_test: int,
        batch_size: int,
        include_inv_permute_idx: bool = False,
    ):
        """Creates a `Batch` from this `SpatiotemporalData`.

        Args:
            rng: A PRNG.
            task: A `SpatiotemporalTask` type, e.g. interpolate or forecast.
            fixed_t_step: If greater than 0, use timesteps separated by a fixed
                step size along the leading axis, `T`, e.g. a `fixed_t_step=2`
                would yield indices 0, 2, 4, 6, ... along the `T` axis. Note
                that this likely only makes sense when the elements in the
                leading axis, `T`, are separated by a fixed time interval, e.g.
                t[0]=1.0, t[1]=2.0, t[2]=3.0, etc. If the time steps along
                the leading axis are real numbers with irregular gaps, e.g.
                t[0]=1.23, t[1]=5.7, t[3]=20.8, etc, this will likely have an
                unintended effect. In summary, `fixed_t_step` works with fixed
                indices along `T`, not with the actual times they represent.
            num_t_per_element: Number of time steps to include in each batch
                element.
            num_ctx_min_per_t: Minimum number of context points per time step.
            num_ctx_max_per_t: Maximum number of context points per time step.
            independent: Whether the subset of context points for each time step
                should be selected independently.
            num_test: Number of test points for target time step.
            batch_size: Number of elements in the generated batch.
            include_inv_permute_idx: Whether to include the `inv_permute_idx`,
                which enables easily mapping context points back to their
                original times and locations in the dataset. This is useful for
                callbacks and plotting functions.
        Returns:
            A `Batch`.
        """
        return _spatiotemporal_data_to_batch(
            rng,
            self.x,
            self.s,
            self.t,
            self.f,
            task,
            fixed_t_step,
            num_t_per_element,
            num_ctx_min_per_t,
            num_ctx_max_per_t,
            independent,
            num_test,
            batch_size,
            include_inv_permute_idx,
        )


# register with jax to use in compiled functions
jax.tree_util.register_pytree_node(
    SpatiotemporalData,
    lambda d: ((d.x, d.s, d.t, d.f), None),
    lambda _aux, children: SpatiotemporalData(*children),
)


@partial(
    jit,
    static_argnames=(
        "task",
        "fixed_t_step",
        "num_t_per_element",
        "num_ctx_min_per_t",
        "num_ctx_max_per_t",
        "independent",
        "num_test",
        "batch_size",
        "include_inv_permute_idx",
    ),
)
def _spatiotemporal_data_to_batch(
    rng: jax.Array,
    x: Optional[jax.Array],  # [T, [S]+, D_x] or [T, 1, D_x] or [1, 1, D_x] or None
    s: jax.Array,  # [T, [S]+, D_s]
    t: jax.Array,  # [T, [S]+, 1] or [T, 1, 1]
    f: jax.Array,  # [T, [S]+, D_f]
    task: SpatiotemporalTask,
    fixed_t_step: int,
    num_t_per_element: int,
    num_ctx_min_per_t: int,
    num_ctx_max_per_t: int,
    independent: bool,
    num_test: int,
    batch_size: int = 4,
    include_inv_permute_idx: bool = False,
):
    B, T, T_e = batch_size, s.shape[0], num_t_per_element
    has_x = x is not None
    flatten_spatial_dims = lambda v: v.reshape(T, -1, v.shape[-1])
    x = flatten_spatial_dims(x) if has_x else None
    s = flatten_spatial_dims(s)
    t = flatten_spatial_dims(t)
    f = flatten_spatial_dims(f)
    L = s.shape[1]
    num_test = num_test or L
    x = jnp.broadcast_to(x, (T, L, x.shape[-1])) if has_x else None
    t = jnp.broadcast_to(t, (T, L, 1))
    rng_t, rng = random.split(rng)
    valid_lens_test = jnp.repeat(num_test, T)
    if fixed_t_step < 0:  # use random time steps
        rng_ts = random.split(rng_t, B)
        ts = vmap(lambda rng: random.choice(rng, T, (T_e,), replace=False))(rng_ts)
        ts = jnp.sort(ts, axis=1)
    else:
        rng_ts = random.split(rng_t, B)
        choices = jnp.arange(fixed_t_step * (num_t_per_element - 1), T)
        last_ts = random.choice(rng, choices, (B,), replace=False)
        prev_ts = fixed_t_step * jnp.arange(num_t_per_element - 1, -1, -1)
        ts = last_ts[:, None] - prev_ts
    # last step if forecast, else median for interpolate
    test_i = -1 if task is SpatiotemporalTask.Forecast else num_t_per_element // 2 + 1
    # get indices of time steps for batches
    t_test_i = ts[:, [test_i]]
    t_ctx_i = jnp.concat([ts[:, :test_i], ts[:, test_i + 1 :]], axis=1)
    x_ctx, x_test = (x[t_ctx_i], x[t_test_i]) if has_x else (None, None)
    # shapes: *_ctx: [B, T_e-1, [S]+, D_*], *_test: [B, 1, [S]+, D_*]
    s_ctx, s_test = s[t_ctx_i], s[t_test_i]
    t_ctx, t_test = t[t_ctx_i], t[t_test_i]
    f_ctx, f_test = f[t_ctx_i], f[t_test_i]
    # TODO:
    # permute each T_e
    # pack the context into a single array
    rngs = random.split(rng, T_e) if independent else jnp.repeat(rng, T_e)
    x_ctx, x_test, _ = vbatch(rngs, x) if has_x else (None, None, None)
    s_ctx, s_test, _ = vbatch(rngs, s)
    t_ctx, t_test, _ = vbatch(rngs, t)
    f_ctx, f_test, inv_permute_idx = vbatch(rngs, f)
    return Batch(
        x_ctx,
        s_ctx,
        t_ctx,
        f_ctx,
        valid_lens_ctx,
        x_test,
        s_test,
        t_test,
        f_test,
        valid_lens_test,
        # no need to store this if its not used
        inv_permute_idx if include_inv_permute_idx else None,
    )
