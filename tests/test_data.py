from time import time

import jax.numpy as jnp
from jax import random
from sps.utils import build_grid

from dl4bi.core.data import BatchElement, Data, SpatialData, TemporalData


def test_data():
    B, L, D = 4, 32, 8
    min_ctx, max_ctx = 4, 16
    rng = random.key(42)
    rng_x, rng_f = random.split(rng)
    x = random.normal(rng_x, (B, L, D))
    f = random.normal(rng_f, (B, L, 1))
    data = Data(x, f)
    batch = data.to_batch(
        rng,
        min_ctx,
        max_ctx,
        num_test=L,
        independent=False,
        test_includes_ctx=True,
        include_inv_permute_idx=True,
    )
    assert batch.x_ctx.shape == (B, max_ctx, D), "Incorrect x_ctx shape!"
    assert batch.f_ctx.shape == (B, max_ctx, 1), "Incorrect f_ctx shape!"
    assert batch.x_test.shape == (B, L, D), "Incorrect x_test shape!"
    assert batch.f_test.shape == (B, L, 1), "Incorrect f_test shape!"
    assert batch.valid_lens_ctx.shape == (B,), "Incorrect valid_lens_ctx shape!"
    assert batch.valid_lens_test.shape == (B,), "Incorrect valid_lens_test shape!"
    assert batch.inv_permute_idx.shape == (B, L), "Incorrect inv_permute_idx shape!"


def test_spatial_data():
    B, S, D = 4, 16, 2
    min_ctx, max_ctx = 8, 64
    rng = random.key(42)
    x = f = random.normal(rng, (B, S, S, D))
    s = build_grid([{"start": -2, "stop": 2, "num": S}] * 2)
    s = jnp.repeat(s[None, ...], B, axis=0)
    data = SpatialData(x, s, f)
    batch = data.to_batch(
        rng,
        min_ctx,
        max_ctx,
        num_test=S * S,
        independent=False,
        test_includes_ctx=True,
        include_inv_permute_idx=True,
    )
    assert batch.x_ctx.shape == (B, max_ctx, D), "Incorrect x_ctx shape!"
    assert batch.s_ctx.shape == (B, max_ctx, D), "Incorrect s_ctx shape!"
    assert batch.f_ctx.shape == (B, max_ctx, D), "Incorrect f_ctx shape!"
    assert batch.x_test.shape == (B, S * S, D), "Incorrect x_test shape!"
    assert batch.s_test.shape == (B, S * S, D), "Incorrect s_test shape!"
    assert batch.f_test.shape == (B, S * S, D), "Incorrect f_test shape!"
    assert batch.valid_lens_ctx.shape == (B,), "Incorrect valid_lens_ctx shape!"
    assert batch.valid_lens_test.shape == (B,), "Incorrect valid_lens_test shape!"
    assert batch.inv_permute_idx.shape == (
        B,
        S * S,
    ), "Incorrect inv_permute_idx shape!"
    num_test = 128
    batch = data.to_batch(
        rng,
        min_ctx,
        max_ctx,
        num_test=num_test,
        independent=False,
        test_includes_ctx=True,
        include_inv_permute_idx=True,
    )
    assert batch.x_ctx.shape == (B, max_ctx, D), "Incorrect x_ctx shape!"
    assert batch.s_ctx.shape == (B, max_ctx, D), "Incorrect s_ctx shape!"
    assert batch.f_ctx.shape == (B, max_ctx, D), "Incorrect f_ctx shape!"
    assert batch.x_test.shape == (B, num_test, D), "Incorrect x_test shape!"
    assert batch.s_test.shape == (B, num_test, D), "Incorrect s_test shape!"
    assert batch.f_test.shape == (B, num_test, D), "Incorrect f_test shape!"
    assert batch.valid_lens_ctx.shape == (B,), "Incorrect valid_lens_ctx shape!"
    assert batch.valid_lens_test.shape == (B,), "Incorrect valid_lens_test shape!"
    assert batch.inv_permute_idx.shape == (
        B,
        S * S,
    ), "Incorrect inv_permute_idx shape!"
    batch = data.to_batch(
        rng,
        min_ctx,
        max_ctx,
        num_test=num_test,
        independent=True,
        test_includes_ctx=False,
        include_inv_permute_idx=True,
    )
    assert batch.x_test.shape == (B, num_test, D), "Incorrect x_test shape!"
    assert batch.s_test.shape == (B, num_test, D), "Incorrect s_test shape!"
    assert batch.f_test.shape == (B, num_test, D), "Incorrect f_test shape!"
    assert batch.inv_permute_idx.shape == (
        B,
        S * S,
    ), "Incorrect inv_permute_idx shape!"
    assert not (batch.inv_permute_idx[0] == batch.inv_permute_idx[1]).all()
    assert isinstance(batch[2], BatchElement), "Indexing not working on batches!"
    be = batch[2]
    assert be.x_ctx.shape == (max_ctx, D), "Incorrect x_ctx shape!"
    assert be.s_ctx.shape == (max_ctx, D), "Incorrect s_ctx shape!"
    assert be.f_ctx.shape == (max_ctx, D), "Incorrect f_ctx shape!"
    assert be.x_test.shape == (num_test, D), "Incorrect x_test shape!"
    assert be.s_test.shape == (num_test, D), "Incorrect s_test shape!"
    assert be.f_test.shape == (num_test, D), "Incorrect f_test shape!"
    assert be.valid_lens_ctx.shape == (), "Incorrect valid_lens_ctx shape!"
    assert be.valid_lens_test.shape == (), "Incorrect valid_lens_test shape!"
    assert be.inv_permute_idx.shape == (S * S,), "Incorrect inv_permute_idx shape!"


def test_temporal_data():
    B, T, D = 4, 128, 2
    min_ctx, max_ctx = 8, 64
    rng = random.key(42)
    x = f = random.normal(rng, (B, T, D))
    t = jnp.repeat(jnp.arange(T)[None, :, None], B, axis=0)
    data = TemporalData(x, t, f)
    batch = data.to_batch(
        rng,
        min_ctx,
        max_ctx,
        num_test=T,
        independent=False,
        test_includes_ctx=True,
        include_inv_permute_idx=True,
    )
    assert batch.x_ctx.shape == (B, max_ctx, D), "Incorrect x_ctx shape!"
    assert batch.f_ctx.shape == (B, max_ctx, D), "Incorrect f_ctx shape!"
    assert batch.x_test.shape == (B, T, D), "Incorrect x_test shape!"
    assert batch.f_test.shape == (B, T, D), "Incorrect f_test shape!"
    assert batch.valid_lens_ctx.shape == (B,), "Incorrect valid_lens_ctx shape!"
    assert batch.valid_lens_test.shape == (B,), "Incorrect valid_lens_test shape!"
    assert batch.inv_permute_idx.shape == (B, T), "Incorrect inv_permute_idx shape!"


def test_speed_without_inv_permute_idx():
    B, S, D, N = 4, 16, 2, 100
    min_ctx, max_ctx = 8, 64
    rng = random.key(42)
    x = f = random.normal(rng, (B, S, S, D))
    s = build_grid([{"start": -2, "stop": 2, "num": S}] * 2)
    s = jnp.repeat(s[None, ...], B, axis=0)
    data = SpatialData(x, s, f)
    # precompile
    data.to_batch(
        rng,
        min_ctx,
        max_ctx,
        num_test=S * S,
        independent=False,
        test_includes_ctx=True,
        include_inv_permute_idx=True,
    )
    t_start = time()
    for i in range(N):
        data.to_batch(
            rng,
            min_ctx,
            max_ctx,
            num_test=S * S,
            independent=False,
            test_includes_ctx=True,
            include_inv_permute_idx=True,
        )
    t_with = time() - t_start
    # precompile
    data.to_batch(
        rng,
        min_ctx,
        max_ctx,
        num_test=S * S,
        independent=False,
        test_includes_ctx=True,
        include_inv_permute_idx=False,
    )
    t_start = time()
    for i in range(N):
        data.to_batch(
            rng,
            min_ctx,
            max_ctx,
            num_test=S * S,
            independent=False,
            test_includes_ctx=True,
            include_inv_permute_idx=False,
        )
    t_without = time() - t_start
    assert t_with > t_without, "No time difference!"
