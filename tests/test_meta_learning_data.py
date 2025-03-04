import jax.numpy as jnp
from jax import random
from sps.utils import build_grid

from dl4bi.meta_learning.data.spatial import SpatialData
from dl4bi.meta_learning.data.tabular import TabularData
from dl4bi.meta_learning.data.temporal import TemporalData


def test_tabular_data():
    rng = random.key(42)
    B, L, D_x, D_f = 4, 37, 8, 1
    num_ctx_min, num_ctx_max, num_test = 3, 10, 20
    x_shape = (B, L, D_x)
    f_shape = (B, L, D_f)
    x_ctx_shape = (B, num_ctx_max, D_x)
    f_ctx_shape = (B, num_ctx_max, D_f)
    mask_ctx_shape = (B, num_ctx_max)
    x_test_shape = (B, num_test, D_x)
    f_test_shape = (B, num_test, D_f)
    mask_test_shape = (B, num_test)
    x = random.normal(rng, x_shape)
    f = random.normal(rng, f_shape)
    # test basic instantiation
    d = TabularData(x, f)
    assert d.x.shape == x_shape
    assert d.f.shape == f_shape
    # test batching where test includes context
    test_includes_ctx = True
    b = d.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.mask_ctx.shape == mask_ctx_shape
    assert b.x_test.shape == x_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.mask_test.shape == mask_test_shape
    assert (b.x_ctx[:, :num_ctx_max] == b.x_test[:, :num_ctx_max]).all()
    assert (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test batching where test does not include context
    test_includes_ctx = False
    b = d.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.mask_ctx.shape == mask_ctx_shape
    assert b.x_test.shape == x_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.mask_test.shape == mask_test_shape
    assert not (b.x_ctx[:, :num_ctx_max] == b.x_test[:, :num_ctx_max]).all()
    assert not (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()


def test_spatial_data_with_x():
    rng = random.key(42)
    B, S, D_x, D_s, D_f = 4, 16, 8, 2, 1
    num_ctx_min, num_ctx_max, num_test = 3, 10, 20
    x_shape = (B, *[S] * D_s, D_x)
    s_shape = (B, *[S] * D_s, D_s)
    f_shape = (B, *[S] * D_s, D_f)
    x_ctx_shape = (B, num_ctx_max, D_x)
    s_ctx_shape = (B, num_ctx_max, D_s)
    f_ctx_shape = (B, num_ctx_max, D_f)
    mask_ctx_shape = (B, num_ctx_max)
    x_test_shape = (B, num_test, D_x)
    s_test_shape = (B, num_test, D_s)
    f_test_shape = (B, num_test, D_f)
    mask_test_shape = (B, num_test)
    x = random.normal(rng, x_shape)
    f = random.normal(rng, f_shape)
    s = build_grid([{"start": -2, "stop": 2, "num": S}] * D_s)
    s = jnp.repeat(s[None, ...], B, axis=0)
    # test basic instantiation
    d = SpatialData(x, s, f)
    assert d.x.shape == x_shape
    assert d.s.shape == s_shape
    assert d.f.shape == f_shape
    # test batching where test includes context
    test_includes_ctx = True
    b = d.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.s_ctx.shape == s_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.mask_ctx.shape == mask_ctx_shape
    assert b.x_test.shape == x_test_shape
    assert b.s_test.shape == s_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.mask_test.shape == mask_test_shape
    assert (b.x_ctx[:, :num_ctx_max] == b.x_test[:, :num_ctx_max]).all()
    assert (b.s_ctx[:, :num_ctx_max] == b.s_test[:, :num_ctx_max]).all()
    assert (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test batching where test does not include context
    test_includes_ctx = False
    b = d.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.s_ctx.shape == s_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.mask_ctx.shape == mask_ctx_shape
    assert b.x_test.shape == x_test_shape
    assert b.s_test.shape == s_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.mask_test.shape == mask_test_shape
    assert not (b.x_ctx[:, :num_ctx_max] == b.x_test[:, :num_ctx_max]).all()
    assert not (b.s_ctx[:, :num_ctx_max] == b.s_test[:, :num_ctx_max]).all()
    assert not (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()


def test_spatial_data_without_x():
    rng = random.key(42)
    B, S, D_s, D_f = 4, 16, 2, 1
    num_ctx_min, num_ctx_max, num_test = 3, 10, 20
    s_shape = (B, *[S] * D_s, D_s)
    f_shape = (B, *[S] * D_s, D_f)
    s_ctx_shape = (B, num_ctx_max, D_s)
    f_ctx_shape = (B, num_ctx_max, D_f)
    mask_ctx_shape = (B, num_ctx_max)
    s_test_shape = (B, num_test, D_s)
    f_test_shape = (B, num_test, D_f)
    mask_test_shape = (B, num_test)
    f = random.normal(rng, f_shape)
    s = build_grid([{"start": -2, "stop": 2, "num": S}] * D_s)
    s = jnp.repeat(s[None, ...], B, axis=0)
    # test basic instantiation
    d = SpatialData(None, s, f)
    assert d.x is None
    assert d.s.shape == s_shape
    assert d.f.shape == f_shape
    # test batching where test includes context
    test_includes_ctx = True
    b = d.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx is None
    assert b.s_ctx.shape == s_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.mask_ctx.shape == mask_ctx_shape
    assert b.x_test is None
    assert b.s_test.shape == s_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.mask_test.shape == mask_test_shape
    assert (b.s_ctx[:, :num_ctx_max] == b.s_test[:, :num_ctx_max]).all()
    assert (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test batching where test does not include context
    test_includes_ctx = False
    b = d.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx is None
    assert b.s_ctx.shape == s_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.mask_ctx.shape == mask_ctx_shape
    assert b.x_test is None
    assert b.s_test.shape == s_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.mask_test.shape == mask_test_shape
    assert not (b.s_ctx[:, :num_ctx_max] == b.s_test[:, :num_ctx_max]).all()
    assert not (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()


def test_spatial_data_broadcast_x():
    rng = random.key(42)
    B, S, D_x, D_s, D_f = 4, 16, 8, 2, 1
    num_ctx_min, num_ctx_max, num_test = 3, 10, 20
    x_shape = (B, D_x)
    s_shape = (B, *[S] * D_s, D_s)
    f_shape = (B, *[S] * D_s, D_f)
    x_ctx_shape = (B, num_ctx_max, D_x)
    s_ctx_shape = (B, num_ctx_max, D_s)
    f_ctx_shape = (B, num_ctx_max, D_f)
    mask_ctx_shape = (B, num_ctx_max)
    x_test_shape = (B, num_test, D_x)
    s_test_shape = (B, num_test, D_s)
    f_test_shape = (B, num_test, D_f)
    mask_test_shape = (B, num_test)
    x = random.normal(rng, x_shape)
    f = random.normal(rng, f_shape)
    s = build_grid([{"start": -2, "stop": 2, "num": S}] * D_s)
    s = jnp.repeat(s[None, ...], B, axis=0)
    # test basic instantiation
    d = SpatialData(x, s, f)
    assert d.x.shape == x_shape
    assert d.s.shape == s_shape
    assert d.f.shape == f_shape
    # test batching where test includes context
    test_includes_ctx = True
    b = d.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.s_ctx.shape == s_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.mask_ctx.shape == mask_ctx_shape
    assert b.x_test.shape == x_test_shape
    assert b.s_test.shape == s_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.mask_test.shape == mask_test_shape
    assert (b.x_ctx[:, :num_ctx_max] == b.x_test[:, :num_ctx_max]).all()
    assert (b.s_ctx[:, :num_ctx_max] == b.s_test[:, :num_ctx_max]).all()
    assert (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test batching where test does not include context
    test_includes_ctx = False
    b = d.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.s_ctx.shape == s_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.mask_ctx.shape == mask_ctx_shape
    assert b.x_test.shape == x_test_shape
    assert b.s_test.shape == s_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.mask_test.shape == mask_test_shape
    assert not (b.s_ctx[:, :num_ctx_max] == b.s_test[:, :num_ctx_max]).all()
    assert not (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()


def test_temporal_data_with_x():
    rng = random.key(42)
    B, T, D_x, D_f = 4, 64, 8, 2
    num_ctx_min, num_ctx_max, num_test = 3, 10, 20
    x_shape = (B, T, D_x)
    t_shape = (B, T)
    f_shape = (B, T, D_f)
    x_ctx_shape = (B, num_ctx_max, D_x)
    t_ctx_shape = (B, num_ctx_max, 1)
    f_ctx_shape = (B, num_ctx_max, D_f)
    mask_ctx_shape = (B, num_ctx_max)
    x_test_shape = (B, num_test, D_x)
    t_test_shape = (B, num_test, 1)
    f_test_shape = (B, num_test, D_f)
    mask_test_shape = (B, num_test)
    x = random.normal(rng, x_shape)
    f = random.normal(rng, f_shape)
    t = jnp.repeat(jnp.arange(T)[None, :], B, axis=0)
    # test basic instantiation
    d = TemporalData(x, t, f)
    assert d.x.shape == x_shape
    assert d.t.shape == t_shape
    assert d.f.shape == f_shape
    # test batching where test includes context
    test_includes_ctx = True
    b = d.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.t_ctx.shape == t_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.mask_ctx.shape == mask_ctx_shape
    assert b.x_test.shape == x_test_shape
    assert b.t_test.shape == t_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.mask_test.shape == mask_test_shape
    assert (b.x_ctx[:, :num_ctx_max] == b.x_test[:, :num_ctx_max]).all()
    assert (b.t_ctx[:, :num_ctx_max] == b.t_test[:, :num_ctx_max]).all()
    assert (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test batching where test does not include context
    test_includes_ctx = False
    b = d.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.t_ctx.shape == t_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.mask_ctx.shape == mask_ctx_shape
    assert b.x_test.shape == x_test_shape
    assert b.t_test.shape == t_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.mask_test.shape == mask_test_shape
    assert not (b.x_ctx[:, :num_ctx_max] == b.x_test[:, :num_ctx_max]).all()
    assert not (b.t_ctx[:, :num_ctx_max] == b.t_test[:, :num_ctx_max]).all()
    assert not (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()


def test_temporal_data_without_x():
    rng = random.key(42)
    B, T, D_f = 4, 64, 2
    num_ctx_min, num_ctx_max, num_test = 3, 10, 20
    t_shape = (B, T)
    f_shape = (B, T, D_f)
    t_ctx_shape = (B, num_ctx_max, 1)
    f_ctx_shape = (B, num_ctx_max, D_f)
    mask_ctx_shape = (B, num_ctx_max)
    t_test_shape = (B, num_test, 1)
    f_test_shape = (B, num_test, D_f)
    mask_test_shape = (B, num_test)
    f = random.normal(rng, f_shape)
    t = jnp.repeat(jnp.arange(T)[None, :], B, axis=0)
    # test basic instantiation
    d = TemporalData(None, t, f)
    assert d.x is None
    assert d.t.shape == t_shape
    assert d.f.shape == f_shape
    # test batching where test includes context
    test_includes_ctx = True
    b = d.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx is None
    assert b.t_ctx.shape == t_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.mask_ctx.shape == mask_ctx_shape
    assert b.x_test is None
    assert b.t_test.shape == t_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.mask_test.shape == mask_test_shape
    assert (b.t_ctx[:, :num_ctx_max] == b.t_test[:, :num_ctx_max]).all()
    assert (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test batching where test does not include context
    test_includes_ctx = False
    b = d.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx is None
    assert b.t_ctx.shape == t_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.mask_ctx.shape == mask_ctx_shape
    assert b.x_test is None
    assert b.t_test.shape == t_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.mask_test.shape == mask_test_shape
    assert not (b.t_ctx[:, :num_ctx_max] == b.t_test[:, :num_ctx_max]).all()
    assert not (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()


def test_temporal_data_broadcast_x():
    rng = random.key(42)
    B, T, D_x, D_f = 4, 64, 8, 2
    num_ctx_min, num_ctx_max, num_test = 3, 10, 20
    x_shape = (B, D_x)
    t_shape = (B, T)
    f_shape = (B, T, D_f)
    x_ctx_shape = (B, num_ctx_max, D_x)
    t_ctx_shape = (B, num_ctx_max, 1)
    f_ctx_shape = (B, num_ctx_max, D_f)
    mask_ctx_shape = (B, num_ctx_max)
    x_test_shape = (B, num_test, D_x)
    t_test_shape = (B, num_test, 1)
    f_test_shape = (B, num_test, D_f)
    mask_test_shape = (B, num_test)
    x = random.normal(rng, x_shape)
    f = random.normal(rng, f_shape)
    t = jnp.repeat(jnp.arange(T)[None, :], B, axis=0)
    # test basic instantiation
    d = TemporalData(x, t, f)
    assert d.x.shape == x_shape
    assert d.t.shape == t_shape
    assert d.f.shape == f_shape
    # test batching where test includes context
    test_includes_ctx = True
    b = d.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.t_ctx.shape == t_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.mask_ctx.shape == mask_ctx_shape
    assert b.x_test.shape == x_test_shape
    assert b.t_test.shape == t_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.mask_test.shape == mask_test_shape
    assert (b.x_ctx[:, :num_ctx_max] == b.x_test[:, :num_ctx_max]).all()
    assert (b.t_ctx[:, :num_ctx_max] == b.t_test[:, :num_ctx_max]).all()
    assert (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()
    # test batching where test does not include context
    test_includes_ctx = False
    b = d.batch(rng, num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    assert b.x_ctx.shape == x_ctx_shape
    assert b.t_ctx.shape == t_ctx_shape
    assert b.f_ctx.shape == f_ctx_shape
    assert b.mask_ctx.shape == mask_ctx_shape
    assert b.x_test.shape == x_test_shape
    assert b.t_test.shape == t_test_shape
    assert b.f_test.shape == f_test_shape
    assert b.mask_test.shape == mask_test_shape
    assert not (b.t_ctx[:, :num_ctx_max] == b.t_test[:, :num_ctx_max]).all()
    assert not (b.f_ctx[:, :num_ctx_max] == b.f_test[:, :num_ctx_max]).all()


def test_spatial_data_plot_1d():
    B, S, D_s, D_f = 4, 128, 1, 1
    num_ctx_min, num_ctx_max, num_test = 3, 10, 128
    rng = random.key(42)
    rng_f, rng_f_pred, rng_b = random.split(rng, 3)
    x = None
    f = random.normal(rng_f, (B, num_test, D_f))
    f_pred = f + 0.01 * random.normal(rng_f_pred, (B, num_test, D_f))
    f_std = jnp.zeros((B, num_test, 1)) + 0.1
    s = build_grid([{"start": -2, "stop": 2, "num": S}] * D_s)
    s = jnp.repeat(s[None, ...], B, axis=0)
    b = SpatialData(x, s, f).batch(rng, num_ctx_min, num_ctx_max, num_test, True)
    fig = b.plot_1d(f_pred, f_std)
    fig.savefig("/tmp/test_spatial_data_plot_1d.png")


def test_temporal_data_plot_1d():
    B, T, D_f = 4, 128, 1
    num_ctx_min, num_ctx_max, num_test = 3, 10, 128
    rng = random.key(42)
    rng_f, rng_f_pred, rng_b = random.split(rng, 3)
    x = None
    f = random.normal(rng_f, (B, num_test, D_f))
    f_pred = f + 0.01 * random.normal(rng_f_pred, (B, num_test, D_f))
    f_std = jnp.zeros((B, num_test, 1)) + 0.1
    t = jnp.repeat(jnp.arange(T)[None, :], B, axis=0)
    b = TemporalData(x, t, f).batch(rng, num_ctx_min, num_ctx_max, num_test, True)
    fig = b.plot_1d(f_pred, f_std)
    fig.savefig("/tmp/test_temporal_data_plot_1d.png")


def test_spatial_data_plot_2d():
    B, S, D_s, D_f = 4, 16, 2, 1
    num_ctx_min, num_ctx_max, num_test = 3, 10, 256
    rng = random.key(42)
    rng_f, rng_f_pred, rng_b = random.split(rng, 3)
    x = None
    f = random.normal(rng_f, (B, num_test, D_f))
    f_pred = f + 0.01 * random.normal(rng_f_pred, (B, num_test, D_f))
    f_std = jnp.zeros((B, num_test, 1)) + 0.1
    s = build_grid([{"start": -2, "stop": 2, "num": S}] * D_s)
    s = jnp.repeat(s[None, ...], B, axis=0)
    b = SpatialData(x, s, f).batch(rng, num_ctx_min, num_ctx_max, num_test, True)
    fig = b.plot_2d(f_pred, f_std)
    fig.savefig("/tmp/test_spatial_data_plot_2d.png")
