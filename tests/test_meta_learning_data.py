import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax.core.frozen_dict import FrozenDict
from jax import jit, random
from sps.sir import LatticeSIR
from sps.utils import build_grid

from dl4bi.meta_learning.data.spatial import SpatialData
from dl4bi.meta_learning.data.spatiotemporal import SpatiotemporalData
from dl4bi.meta_learning.data.tabular import TabularData
from dl4bi.meta_learning.utils import so3_rotate


def test_rotations():
    s = jnp.array([[0, 0]])
    # 90 north, 20 ccw tilt should give -90, 70
    assert jnp.allclose(so3_rotate(s, 90, 0, 20), jnp.array([[-90, 70]]))
    assert jnp.allclose(so3_rotate(s, 0, 0, 0), jnp.array(s))
    assert jnp.allclose(so3_rotate(s, 0, 90, 0), jnp.array([[90, 0]]))
    assert jnp.allclose(so3_rotate(s, 25, 0, 0), jnp.array([[0, 25]]))


def test_tabular_data():
    rng = random.key(42)
    B, L, D_x, D_z, D_t, D_f = 4, 37, 8, 3, 1, 1
    num_ctx_min, num_ctx_max, num_test = 3, 10, 20
    x_shape = (B, L, D_x)
    z_shape = (B, L, D_z)  # another random variable group
    t_shape = (B, L, D_t)
    f_shape = (B, L, D_f)
    x_ctx_shape = (B, num_ctx_max, D_x)
    z_ctx_shape = (B, num_ctx_max, D_z)
    t_ctx_shape = (B, num_ctx_max, D_t)
    f_ctx_shape = (B, num_ctx_max, D_f)
    mask_ctx_shape = (B, num_ctx_max)
    x_test_shape = (B, num_test, D_x)
    z_test_shape = (B, num_test, D_z)
    t_test_shape = (B, num_test, D_t)
    f_test_shape = (B, num_test, D_f)
    mask_test_shape = (B, num_test)
    x = random.normal(rng, x_shape)
    z = random.normal(rng, z_shape)
    f = random.normal(rng, f_shape)
    t = jnp.repeat(jnp.arange(L)[None, :, None], B, axis=0)
    # test basic instantiation
    d = TabularData(FrozenDict({"x": x, "z": z, "t": t}), f)
    assert d.feature_groups["x"].shape == x_shape
    assert d.feature_groups["z"].shape == z_shape
    assert d.feature_groups["t"].shape == t_shape
    assert d.f.shape == f_shape
    # test batching where test includes context
    test_includes_ctx = True
    forecast = False
    t_sorted = True
    b = d.batch(
        rng,
        num_ctx_min,
        num_ctx_max,
        num_test,
        test_includes_ctx,
        forecast,
        t_sorted,
    )
    assert set(b) == {
        "x_ctx",
        "z_ctx",
        "t_ctx",
        "f_ctx",
        "mask_ctx",
        "x_test",
        "z_test",
        "t_test",
        "f_test",
        "mask_test",
        "inv_permute_idx",
    }
    samples = b.sample_for_inference(rng, num_samples=1)
    assert len(samples) == 1
    samples = b.sample_for_inference(rng, num_samples=2)
    assert samples[0][1]["f_ctx"].shape[0] < num_ctx_max
    assert len(samples) == 2
    # assert that they have been masked and extracted differently
    # this could fail if they just so happen to have the same number
    # of valid lengths for each sequence although this is p=1/37
    assert samples[0][1]["f_ctx"].shape != samples[1][1]["f_ctx"].shape
    assert b.ctx["x_ctx"].shape == x_ctx_shape
    assert b.ctx["z_ctx"].shape == z_ctx_shape
    assert b.ctx["t_ctx"].shape == t_ctx_shape
    assert b.ctx["f_ctx"].shape == f_ctx_shape
    assert b.mask_ctx.shape == mask_ctx_shape
    assert b.test["x_test"].shape == x_test_shape
    assert b.test["z_test"].shape == z_test_shape
    assert b.test["t_test"].shape == t_test_shape
    assert b.test["f_test"].shape == f_test_shape
    assert b.mask_test.shape == mask_test_shape
    assert (b.ctx["x_ctx"][:, :num_ctx_max] == b.test["x_test"][:, :num_ctx_max]).all()
    assert (b.ctx["z_ctx"][:, :num_ctx_max] == b.test["z_test"][:, :num_ctx_max]).all()
    assert (b.ctx["t_ctx"][:, :num_ctx_max] == b.test["t_test"][:, :num_ctx_max]).all()
    assert (b.ctx["f_ctx"][:, :num_ctx_max] == b.test["f_test"][:, :num_ctx_max]).all()
    # test batching where test does not include context
    test_includes_ctx = False
    forecast = True
    b = d.batch(
        rng,
        num_ctx_min,
        num_ctx_max,
        num_test,
        test_includes_ctx,
        forecast,
        t_sorted,
    )
    assert b.ctx["x_ctx"].shape == x_ctx_shape
    assert b.ctx["z_ctx"].shape == z_ctx_shape
    assert b.ctx["t_ctx"].shape == t_ctx_shape
    assert b.ctx["f_ctx"].shape == f_ctx_shape
    assert b.mask_ctx.shape == mask_ctx_shape
    assert b.test["x_test"].shape == x_test_shape
    assert b.test["z_test"].shape == z_test_shape
    assert b.test["t_test"].shape == t_test_shape
    assert b.test["f_test"].shape == f_test_shape
    assert b.mask_test.shape == mask_test_shape
    assert not (
        b.ctx["x_ctx"][:, :num_ctx_max] == b.test["x_test"][:, :num_ctx_max]
    ).all()
    assert not (
        b.ctx["z_ctx"][:, :num_ctx_max] == b.test["z_test"][:, :num_ctx_max]
    ).all()
    assert not (
        b.ctx["t_ctx"][:, :num_ctx_max] == b.test["t_test"][:, :num_ctx_max]
    ).all()
    assert not (
        b.ctx["f_ctx"][:, :num_ctx_max] == b.test["f_test"][:, :num_ctx_max]
    ).all()
    assert (b["t_test"].min(axis=1) > b["t_ctx"].max(axis=1)).all()
    t = jnp.repeat(random.permutation(rng, jnp.arange(L))[None, :, None], B, axis=0)
    t_sorted = False
    b = d.batch(
        rng,
        num_ctx_min,
        num_ctx_max,
        num_test,
        test_includes_ctx,
        forecast,
        t_sorted,
    )
    assert (b["t_test"].min(axis=1) > b["t_ctx"].max(axis=1)).all()


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
    samples = b.sample_for_inference(rng, num_samples=1)
    assert len(samples) == 1
    samples = b.sample_for_inference(rng, num_samples=2)
    assert samples[0][1]["f_ctx"].shape[0] < num_ctx_max
    assert len(samples) == 2
    # assert that they have been masked and extracted differently
    # this could fail if they just so happen to have the same number
    # of valid lengths for each sequence although this is p=1/37
    assert samples[0][1]["f_ctx"].shape != samples[1][1]["f_ctx"].shape
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
    samples = b.sample_for_inference(rng, num_samples=1)
    assert len(samples) == 1
    samples = b.sample_for_inference(rng, num_samples=2)
    assert samples[0][1]["f_ctx"].shape[0] < num_ctx_max
    assert len(samples) == 2
    # assert that they have been masked and extracted differently
    # this could fail if they just so happen to have the same number
    # of valid lengths for each sequence although this is p=1/37
    assert samples[0][1]["f_ctx"].shape != samples[1][1]["f_ctx"].shape
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
    samples = b.sample_for_inference(rng, num_samples=1)
    assert len(samples) == 1
    samples = b.sample_for_inference(rng, num_samples=2)
    assert samples[0][1]["f_ctx"].shape[0] < num_ctx_max
    assert len(samples) == 2
    # assert that they have been masked and extracted differently
    # this could fail if they just so happen to have the same number
    # of valid lengths for each sequence although this is p=1/37
    assert samples[0][1]["f_ctx"].shape != samples[1][1]["f_ctx"].shape
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


def test_spatiotemporal_data_with_x():
    rng = random.key(42)
    B, T, T_b, S, D_x, D_s, D_t, D_f = 4, 7, 5, 16, 8, 2, 1, 1
    num_ctx_min_per_t, num_ctx_max_per_t, num_test = 3, 10, 20
    x_shape = (T, *[S] * D_s, D_x)
    s_shape = (T, *[S] * D_s, D_s)
    t_shape = (T,)
    f_shape = (T, *[S] * D_s, D_f)
    x_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1), D_x)
    s_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1), D_s)
    t_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1), D_t)
    f_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1), D_f)
    mask_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1))
    x_test_shape = (B, num_test, D_x)
    s_test_shape = (B, num_test, D_s)
    t_test_shape = (B, num_test, D_t)
    f_test_shape = (B, num_test, D_f)
    mask_test_shape = (B, num_test)
    x = random.normal(rng, x_shape)
    f = random.normal(rng, f_shape)
    s = build_grid([{"start": -2, "stop": 2, "num": S}] * D_s)
    s = jnp.repeat(s[None, ...], T, axis=0)
    t = jnp.arange(T)
    # test basic instantiation
    d = SpatiotemporalData(x, s, t, f)
    assert d.x.shape == x_shape
    assert d.s.shape == s_shape
    assert d.t.shape == t_shape
    assert d.f.shape == f_shape
    for forecast in [True, False]:
        for random_t in [True, False]:
            for independent_t_masks in [True, False]:
                b = d.batch(
                    rng,
                    T_b,
                    random_t,
                    num_ctx_min_per_t,
                    num_ctx_max_per_t,
                    independent_t_masks,
                    num_test,
                    forecast,
                    B,
                )
                samples = b.sample_for_inference(rng, num_samples=1)
                assert len(samples) == 1
                assert samples[0][1]["f_ctx"].shape[0] < num_ctx_max_per_t * (T_b - 1)
                samples = b.sample_for_inference(rng, num_samples=2)
                assert len(samples) == 2
                # assert that they have been masked and extracted differently
                # this could fail if they just so happen to have the same number
                # of valid lengths for each sequence although this is p=1/37
                assert b.x_ctx.shape == x_ctx_shape
                assert b.s_ctx.shape == s_ctx_shape
                assert b.t_ctx.shape == t_ctx_shape
                assert b.f_ctx.shape == f_ctx_shape
                assert b.mask_ctx.shape == mask_ctx_shape
                assert b.x_test.shape == x_test_shape
                assert b.s_test.shape == s_test_shape
                assert b.t_test.shape == t_test_shape
                assert b.f_test.shape == f_test_shape
                assert b.mask_test.shape == mask_test_shape


def test_spatiotemporal_data_without_x():
    rng = random.key(42)
    B, T, T_b, S, D_s, D_t, D_f = 4, 7, 5, 16, 2, 1, 1
    num_ctx_min_per_t, num_ctx_max_per_t, num_test = 3, 10, 20
    s_shape = (T, *[S] * D_s, D_s)
    t_shape = (T,)
    f_shape = (T, *[S] * D_s, D_f)
    s_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1), D_s)
    t_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1), D_t)
    f_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1), D_f)
    mask_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1))
    s_test_shape = (B, num_test, D_s)
    t_test_shape = (B, num_test, D_t)
    f_test_shape = (B, num_test, D_f)
    mask_test_shape = (B, num_test)
    f = random.normal(rng, f_shape)
    s = build_grid([{"start": -2, "stop": 2, "num": S}] * D_s)
    s = jnp.repeat(s[None, ...], T, axis=0)
    t = jnp.arange(T)
    # test basic instantiation
    d = SpatiotemporalData(None, s, t, f)
    assert d.x is None
    assert d.s.shape == s_shape
    assert d.t.shape == t_shape
    assert d.f.shape == f_shape
    for forecast in [True, False]:
        for random_t in [True, False]:
            for independent_t_masks in [True, False]:
                b = d.batch(
                    rng,
                    T_b,
                    random_t,
                    num_ctx_min_per_t,
                    num_ctx_max_per_t,
                    independent_t_masks,
                    num_test,
                    forecast,
                    B,
                )
                print(forecast, random_t, independent_t_masks)
                assert b.x_ctx is None
                assert b.s_ctx.shape == s_ctx_shape
                assert b.t_ctx.shape == t_ctx_shape
                assert b.f_ctx.shape == f_ctx_shape
                assert b.mask_ctx.shape == mask_ctx_shape
                assert b.x_test is None
                assert b.s_test.shape == s_test_shape
                assert b.t_test.shape == t_test_shape
                assert b.f_test.shape == f_test_shape
                assert b.mask_test.shape == mask_test_shape


def test_spatiotemporal_data_broadcast_x():
    rng = random.key(42)
    B, T, T_b, S, D_x, D_s, D_t, D_f = 4, 7, 5, 16, 8, 2, 1, 1
    num_ctx_min_per_t, num_ctx_max_per_t, num_test = 3, 10, 20
    x_shape = (T, D_x)
    s_shape = (T, *[S] * D_s, D_s)
    t_shape = (T,)
    f_shape = (T, *[S] * D_s, D_f)
    x_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1), D_x)
    s_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1), D_s)
    t_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1), D_t)
    f_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1), D_f)
    mask_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1))
    x_test_shape = (B, num_test, D_x)
    s_test_shape = (B, num_test, D_s)
    t_test_shape = (B, num_test, D_t)
    f_test_shape = (B, num_test, D_f)
    mask_test_shape = (B, num_test)
    x = random.normal(rng, x_shape)
    f = random.normal(rng, f_shape)
    s = build_grid([{"start": -2, "stop": 2, "num": S}] * D_s)
    s = jnp.repeat(s[None, ...], T, axis=0)
    t = jnp.arange(T)
    # test basic instantiation
    d = SpatiotemporalData(x, s, t, f)
    assert d.x.shape == x_shape
    assert d.s.shape == s_shape
    assert d.t.shape == t_shape
    assert d.f.shape == f_shape
    for forecast in [True, False]:
        for random_t in [True, False]:
            for independent_t_masks in [True, False]:
                b = d.batch(
                    rng,
                    T_b,
                    random_t,
                    num_ctx_min_per_t,
                    num_ctx_max_per_t,
                    independent_t_masks,
                    num_test,
                    forecast,
                    B,
                )
                assert b.x_ctx.shape == x_ctx_shape
                assert b.s_ctx.shape == s_ctx_shape
                assert b.t_ctx.shape == t_ctx_shape
                assert b.f_ctx.shape == f_ctx_shape
                assert b.mask_ctx.shape == mask_ctx_shape
                assert b.x_test.shape == x_test_shape
                assert b.s_test.shape == s_test_shape
                assert b.t_test.shape == t_test_shape
                assert b.f_test.shape == f_test_shape
                assert b.mask_test.shape == mask_test_shape


def test_spatiotemporal_data_broadcast_single_x():
    rng = random.key(42)
    B, T, T_b, S, D_x, D_s, D_t, D_f = 4, 7, 5, 16, 8, 2, 1, 1
    num_ctx_min_per_t, num_ctx_max_per_t, num_test = 3, 10, 20
    x_shape = (D_x,)
    s_shape = (T, *[S] * D_s, D_s)
    t_shape = (T,)
    f_shape = (T, *[S] * D_s, D_f)
    x_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1), D_x)
    s_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1), D_s)
    t_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1), D_t)
    f_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1), D_f)
    mask_ctx_shape = (B, num_ctx_max_per_t * (T_b - 1))
    x_test_shape = (B, num_test, D_x)
    s_test_shape = (B, num_test, D_s)
    t_test_shape = (B, num_test, D_t)
    f_test_shape = (B, num_test, D_f)
    mask_test_shape = (B, num_test)
    x = random.normal(rng, x_shape)
    f = random.normal(rng, f_shape)
    s = build_grid([{"start": -2, "stop": 2, "num": S}] * D_s)
    s = jnp.repeat(s[None, ...], T, axis=0)
    t = jnp.arange(T)
    # test basic instantiation
    d = SpatiotemporalData(x, s, t, f)
    assert d.x.shape == x_shape
    assert d.s.shape == s_shape
    assert d.t.shape == t_shape
    assert d.f.shape == f_shape
    for forecast in [True, False]:
        for random_t in [True, False]:
            for independent_t_masks in [True, False]:
                b = d.batch(
                    rng,
                    T_b,
                    random_t,
                    num_ctx_min_per_t,
                    num_ctx_max_per_t,
                    independent_t_masks,
                    num_test,
                    forecast,
                    B,
                )
                assert b.x_ctx.shape == x_ctx_shape
                assert b.s_ctx.shape == s_ctx_shape
                assert b.t_ctx.shape == t_ctx_shape
                assert b.f_ctx.shape == f_ctx_shape
                assert b.mask_ctx.shape == mask_ctx_shape
                assert b.x_test.shape == x_test_shape
                assert b.s_test.shape == s_test_shape
                assert b.t_test.shape == t_test_shape
                assert b.f_test.shape == f_test_shape
                assert b.mask_test.shape == mask_test_shape


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


def test_spatiotemporal_data_plot_2d():
    B, T, T_b, S = 4, 25, 5, 64
    num_test = 64 * 64
    num_ctx_min_per_t, num_ctx_max_per_t = int(num_test * 0.5), int(num_test * 0.75)
    rng = random.key(42)
    sir = LatticeSIR()
    f, *_ = sir.simulate(rng, (S, S), T)
    f = rsi_to_rgb(f)
    s = build_grid([{"start": -2, "stop": 2, "num": S}] * 2)
    s = jnp.repeat(s[None, ...], T, axis=0)
    t = jnp.arange(T)
    d = SpatiotemporalData(None, s, t, f)
    _, axs = plt.subplots(T, 1, figsize=(5, B * 5))
    for forecast in [True, False]:
        for random_t in [True, False]:
            for independent_t_masks in [True, False]:
                b = d.batch(
                    rng,
                    T_b,
                    random_t,
                    num_ctx_min_per_t,
                    num_ctx_max_per_t,
                    independent_t_masks,
                    num_test,
                    forecast,
                    B,
                )
                f_pred = b.f_test + 0.01 * random.normal(rng, b.f_test.shape)
                f_std = jnp.zeros(b.f_test.shape) + 0.1
                fig = b.plot_2d(f_pred, f_std, remap_colors=remap_colors)
                suffix = f"forecast_{forecast}_random_t_{random_t}_independent_t_masks_{independent_t_masks}"
                fig.savefig(f"/tmp/test_spatiotemporal_data_plot_2d_{suffix}.png")
                plt.close(fig)


@jit
def rsi_to_rgb(steps: jax.Array):
    steps += 1  # [-1, 0, 1] -> [0, 1, 2]
    # 0 (recovered) => 1 (green)
    # 1 (susceptible) => 2 (blue)
    # 2 (infected) => 0 (red)
    mapping = jnp.array([1, 2, 0])
    rgb_cat = mapping[jnp.int32(steps)]
    # convert RGB categories to one-hot vectors
    return jax.nn.one_hot(rgb_cat, 3)


@jit
def remap_colors(x: jax.Array):
    # palette from https://davidmathlogic.com/colorblind
    C = jnp.array([[216, 27, 96], [0, 77, 64], [30, 136, 229]]) / 255.0
    C = C[None, None, None, ...]
    return (C * x[..., None]).sum(axis=-2)
