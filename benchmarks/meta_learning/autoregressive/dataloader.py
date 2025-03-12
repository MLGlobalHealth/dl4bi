import jax
import jax.numpy as jnp
from jax import jit, random
from omegaconf import DictConfig

from dl4bi.meta_learning.train_utils import build_grid, instantiate


def build_gp_dataloader(data: DictConfig, kernel: DictConfig):
    """Generates batches of GP observations.

    Note that it differs from the dataloader found in `gp.py`
    in that it yields `f_test` with observation noise.
    """
    gp = instantiate(kernel)
    B, S = data.batch_size, len(data.s)
    Nc_min, Nc_max = data.num_ctx.min, data.num_ctx.max
    s_g = build_grid(data.s).reshape(-1, S)  # flatten spatial dims
    L = Nc_max + s_g.shape[0]  # L = num test or all points
    obs_noise, B = data.obs_noise, data.batch_size
    valid_lens_test = jnp.repeat(L, B)
    s_min = jnp.array([axis["start"] for axis in data.s])
    s_max = jnp.array([axis["stop"] for axis in data.s])
    batchify = jit(lambda x: jnp.repeat(x[None, ...], B, axis=0))

    def gen_batch(rng: jax.Array):
        rng_s, rng_gp, rng_v, rng_ctx, rng_test = random.split(rng, 5)
        s_r = random.uniform(rng_s, (Nc_max, S), jnp.float32, s_min, s_max)
        s = jnp.vstack([s_r, s_g])
        f, var, ls, period, *_ = gp.simulate(rng_gp, s, B)
        valid_lens_ctx = random.randint(rng_v, (B,), Nc_min, Nc_max)
        s = batchify(s)
        s_ctx = s[:, :Nc_max, :]
        f_ctx = f + obs_noise * random.normal(rng_ctx, f.shape)
        f_ctx = f_ctx[:, :Nc_max, :]
        s_test = s
        f_test = f
        f_test_obs = f_test + obs_noise * random.normal(rng_test, f.shape)
        return (
            s_ctx,
            f_ctx,
            valid_lens_ctx,
            s_test,
            f_test,
            f_test_obs,
            valid_lens_test,
            var,
            ls,
            period,
        )

    def dataloader(rng: jax.Array):
        while True:
            rng_i, rng = random.split(rng)
            yield gen_batch(rng_i)

    return dataloader
