import jax.numpy as jnp
from jax import random
from sps.utils import build_grid

from dl4bi.core.data import BatchElement, SpatialData


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
        num_test=None,
        independent=False,
        test_includes_ctx=True,
    )
    assert batch.x_ctx.shape == (B, max_ctx, D), "Incorrect x_ctx shape!"
    assert batch.s_ctx.shape == (B, max_ctx, D), "Incorrect s_ctx shape!"
    assert batch.f_ctx.shape == (B, max_ctx, D), "Incorrect f_ctx shape!"
    assert batch.x_test.shape == (B, S * S, D), "Incorrect x_test shape!"
    assert batch.s_test.shape == (B, S * S, D), "Incorrect s_test shape!"
    assert batch.f_test.shape == (B, S * S, D), "Incorrect f_test shape!"
    assert batch.valid_lens_ctx.shape == (B,), "Incorrect valid_lens_ctx shape!"
    assert batch.valid_lens_test.shape == (B,), "Incorrect valid_lens_test shape!"
    assert batch.inv_permute_idx.shape == (S * S,), "Incorrect inv_permute_idx shape!"
    num_test = 128
    batch = data.to_batch(
        rng,
        min_ctx,
        max_ctx,
        num_test=num_test,
        independent=False,
        test_includes_ctx=True,
    )
    assert batch.x_ctx.shape == (B, max_ctx, D), "Incorrect x_ctx shape!"
    assert batch.s_ctx.shape == (B, max_ctx, D), "Incorrect s_ctx shape!"
    assert batch.f_ctx.shape == (B, max_ctx, D), "Incorrect f_ctx shape!"
    assert batch.x_test.shape == (B, num_test, D), "Incorrect x_test shape!"
    assert batch.s_test.shape == (B, num_test, D), "Incorrect s_test shape!"
    assert batch.f_test.shape == (B, num_test, D), "Incorrect f_test shape!"
    assert batch.valid_lens_ctx.shape == (B,), "Incorrect valid_lens_ctx shape!"
    assert batch.valid_lens_test.shape == (B,), "Incorrect valid_lens_test shape!"
    assert batch.inv_permute_idx.shape == (S * S,), "Incorrect inv_permute_idx shape!"
    batch = data.to_batch(
        rng,
        min_ctx,
        max_ctx,
        num_test=num_test,
        independent=True,
        test_includes_ctx=False,
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
