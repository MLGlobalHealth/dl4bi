import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from numpyro import distributions as dist
from sps.utils import build_grid
from sps.kernels import matern_3_2
from jax import jit

priors = {"ls": dist.Uniform(0, 100)}


@partial(jit, static_argnames=["max_s", "min_locations", "max_locations"])
def sample_uniform_s(rng, max_s, min_locations, max_locations):
    rng_loc, rng_s = random.split(rng)
    s = random.uniform(rng_s, (max_locations, 2), maxval=max_s)
    num_locs = max_locations
    if min_locations < max_locations:
        num_locs = random.randint(
            rng_loc, shape=tuple(), minval=min_locations, maxval=max_locations + 1
        )
    return s, num_locs


def gen_dataloader(
    grid_size, priors, kernel, max_s, batch_size=32, uniform_s=True, min_locations=None
):
    jitter = 5e-4 * jnp.eye(grid_size)
    if uniform_s:
        s_jit = partial(
            sample_uniform_s,
            max_s=max_s,
            min_locations=grid_size if min_locations is None else min_locations,
            max_locations=grid_size,
        )
    else:
        s_fixed = build_grid(
            [{"start": 0.0, "stop": max_s, "num": int(jnp.sqrt(grid_size))}] * 2
        ).reshape(-1, 2)
        s_jit = jit(lambda rng: s_fixed, grid_size)
    kernel_jit = jit(lambda s, var, ls: kernel(s, s, var, ls) + jitter)
    f_jit = jit(lambda K, z: jnp.einsum("ij,bj->bi", jnp.linalg.cholesky(K), z))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z, rng_samp = random.split(rng_data, 4)
            var = 1.0
            s, num_locs = s_jit(rng_samp)
            ls = priors["ls"].sample(rng_ls)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, grid_size))
            K = kernel_jit(s, var, ls)
            f = f_jit(K, z.at[:, num_locs:].set(0))
            mask = jnp.repeat(
                (jnp.arange(grid_size) < num_locs)[None], batch_size, axis=0
            )
            yield {
                "s": s,
                "f": f,
                "z": z,
                "mask": mask,
                "conditionals": jnp.array([ls]),
                "K": K,
            }

    return dataloader


# # Test parameters
# grid_size = 256
# min_locations = 64
# batch_size = 4
# num_seeds = 10

# dataloader_fn = gen_dataloader(
#     grid_size=grid_size,
#     priors=priors,
#     kernel=matern_3_2,
#     max_s=1.0,
#     batch_size=batch_size,
#     uniform_s=True,
#     min_locations=min_locations,
# )

# # Loop over seeds
# for seed in range(num_seeds):
#     rng = random.PRNGKey(seed)
#     dl = dataloader_fn(rng)
#     batch = next(dl)

#     f = batch["f"]
#     mask = batch["mask"]
#     num_locs = jnp.sum(mask[0])  # all rows have same num_locs

#     # Check mask
#     assert jnp.all(mask[:, :num_locs] == 1) and jnp.all(mask[:, num_locs:] == 0), (
#         "mask incorrect"
#     )

#     # Check top-left Cholesky reconstruction
#     L = jnp.linalg.cholesky(batch["K"][:num_locs, :num_locs])
#     reconstructed = jnp.einsum("ij,bj->bi", L, batch["z"][:, :num_locs])
#     assert jnp.allclose(reconstructed, f[:, :num_locs]), (
#         "f not equal to L @ z for top-left block"
#     )

#     print(f"Seed {seed} passed. num_locs = {num_locs}")
import jax
import jax.numpy as jnp


def test_preconditioning():
    key = jax.random.PRNGKey(0)
    N, U = 5, 3
    s = jax.random.normal(key, (N, 2))  # N points
    u = jax.random.normal(key, (U, 2))  # U inducing points
    var, ls = 1.0, 0.7

    # toy kernel
    def matern_1_2(x, y, var, ls):
        d = jnp.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
        return var * jnp.exp(-d / ls)

    K_su = matern_1_2(s, u, var, ls)
    f_u_bar = jax.random.normal(key, (U,))

    # reference preconditioning D
    D = jnp.linalg.norm(matern_1_2(u, u, var, ls), axis=0)
    D_inv = 1.0 / D

    # --- 1. check equivalence of multiplications ---
    f_scaled = f_u_bar * D  # should equal D @ f_u_bar
    f_matmul = jnp.diag(D) @ f_u_bar
    print("‖f_scaled - f_matmul‖:", jnp.linalg.norm(f_scaled - f_matmul))

    K_scaled = K_su * D_inv  # should equal K_su @ D_inv
    K_matmul = K_su @ jnp.diag(D_inv)
    print("‖K_scaled - K_matmul‖:", jnp.linalg.norm(K_scaled - K_matmul))

    # --- 2. check reconstruction of f ---
    f_true = K_su @ f_u_bar
    f_pre = K_scaled @ f_scaled
    print("‖f_true - f_pre‖:", jnp.linalg.norm(f_true - f_pre))


test_preconditioning()
