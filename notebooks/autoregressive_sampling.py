from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import jit, random
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dl4bi.meta_learning.autoregressive import (
    build_gp_dataloader,
    closest_first,
    furthest_first,
    invert_permutation,
    sample_autoreg,
    sample_diagonal,
)
from dl4bi.meta_learning.train_utils import load_ckpt

device = jax.devices()[0]


def load_model(path):
    if not Path("results").exists():
        Path("results").hardlink_to(path)

    model_dir = Path("results")
    models = sorted(list(model_dir.glob("**/*.ckpt")), key=lambda x: x.stat().st_mtime)

    prompt = (
        "Select model to load:\n"
        + "\n".join(f"{i}: {model}" for i, model in enumerate(models))
        + "\n"
    )
    i = int(input(prompt))
    model_path = models[i]

    state, config = load_ckpt(model_path)

    def apply(
        s_ctx: jax.Array,
        f_ctx: jax.Array,
        s_test: jax.Array,
        valid_lens_ctx: Optional[jax.Array] = None,
    ):
        return state.apply_fn(
            {"params": state.params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            training=False,
            # rngs={"extra": rng_extra}, # what happens if this is omitted? is this used test-time?
        )

    return jit(apply), config


def run(
    *,
    apply: Callable,
    config: OmegaConf,
    seed: int,
    job_id: Optional[str] = None,
    Np=32,  # number of paths to sample autoregressively, will be rounded up to be a multiple of batch size
    B=32,  # how to batch the autoregressive sampling. this parameter only affects runtime efficiency
    ls: float | None = None,  # override the ls in the config
    var: float | None = None,  # override the var in the config
    num_ctx: int | None = None,  # number of context points to use
    noise: float | None = None,  # override the noise in the config
    debug: bool = False,
):
    rng = random.key(seed)

    if not job_id:
        job_id = str(datetime.now())
    results_dir = Path(f"results/{job_id}")
    results_dir.mkdir(parents=True)

    jnp.save(results_dir / "seed.npy", seed)

    rng, rng_dataloader = jax.random.split(rng)

    config = config.copy()
    config.data.batch_size = 1
    if ls is not None:
        config.kernel.kwargs.ls.kwargs = {"dist": "fixed", "kwargs": {"value": ls}}
    if var is not None:
        config.kernel.kwargs.var.kwargs = {"dist": "fixed", "kwargs": {"value": var}}

    if num_ctx is not None:
        config.data.num_ctx = {"min": num_ctx, "max": num_ctx}
    if noise is not None:
        config.data.obs_noise = noise

    OmegaConf.save(config, results_dir / "config.yaml")
    print("Seed:", seed)
    print("GP:", config.kernel)
    print("Data:", config.data)

    dataloader = build_gp_dataloader(config.data, config.kernel)(rng_dataloader)
    # note: unpacking the size-1 batch
    (
        [s_ctx],
        [f_ctx],
        [valid_lens_ctx],
        [s],
        [f],
        _valid_lens_test,  # unused
        [var],
        [ls],
        _period,  # unused
    ) = next(dataloader)
    print(f"var {var}, ls {ls}")
    jnp.save(results_dir / "var.npy", var)
    jnp.save(results_dir / "ls.npy", ls)

    f_ctx = f_ctx[:valid_lens_ctx]
    s_ctx = s_ctx[:valid_lens_ctx]

    # Excluding the context points from test set for numerical stability of later analysis.
    num_ctx = config.data.num_ctx.max
    s_test = s[num_ctx:]
    f_test = f[num_ctx:]

    # Save everything that can be needed for later analysis
    jnp.save(results_dir / "s_ctx.npy", s_ctx)
    jnp.save(results_dir / "f_ctx.npy", f_ctx)
    jnp.save(results_dir / "s_test.npy", s_test)
    jnp.save(results_dir / "f_test.npy", f_test)
    jnp.save(results_dir / "s.npy", s)
    jnp.save(results_dir / "f.npy", f)

    # Regular (diagonal) TNP-KR
    [diagonal_mu], [diagonal_sd] = apply(
        s_ctx[None], f_ctx[None], s_test[None]
    )  # note need to expand dims
    diagonal_mu, diagonal_sd = diagonal_mu.squeeze(), diagonal_sd.squeeze()
    diagonal_var = diagonal_sd**2
    jnp.save(results_dir / "diagonal_mu", diagonal_mu)
    jnp.save(results_dir / "diagonal_var", diagonal_var)

    # Put arrays on gpu explicitly
    s_ctx = jax.device_put(s_ctx, device)
    f_ctx = jax.device_put(f_ctx, device)
    s_test = jax.device_put(s_test, device)

    # Autoregressive paths sampling
    if Np > 0:
        num_iters = (Np - 1) // B + 1
        for strategy in [
            # "preserve",
            "diagonal",
            "ltr",
            "furthest",
            "random",
            "closest",
        ]:
            print(f"Strategy: {strategy}")
            paths, densities = [], []

            # there was significant overhead from reordering within each batch, hence do that here
            match strategy:
                case "diagonal":
                    # perhaps it might make sense to try reordering the diagonal paths as well?
                    # the model should be permutation-invariant though
                    idx = idx_inv = ...
                case "preserve":
                    idx = idx_inv = ...
                case "ltr":
                    idx = jnp.argsort(s_test, axis=None)
                    idx_inv = invert_permutation(idx)
                case "closest":
                    idx = closest_first(s_ctx, s_test)
                    idx_inv = invert_permutation(idx)
                case "furthest":
                    idx = furthest_first(s_ctx, s_test)
                    idx_inv = invert_permutation(idx)
                case "random":
                    # handled inside sample_paths, so identity here
                    idx = idx_inv = ...

            s_test = s_test[idx]

            for i in tqdm(range(num_iters)):
                rng, rng_i = random.split(rng)
                if strategy == "diagonal":
                    path, log_density = sample_diagonal(
                        rng_i, apply, s_ctx, f_ctx, s_test, B
                    )
                else:
                    path, log_density = sample_autoreg(
                        rng_i,
                        apply,
                        s_ctx,
                        f_ctx,
                        s_test,
                        B,
                        random=(strategy == "random"),
                        debug=debug,
                    )
                paths.append(path)
                densities.append(log_density)

            paths = jnp.concat(paths, axis=0)[:, idx_inv]
            jnp.save(results_dir / f"paths_{strategy}.npy", paths)
            densities = jnp.concat(densities, axis=0)
            jnp.save(results_dir / f"densities_{strategy}.npy", densities)


if __name__ == "__main__":
    from argparse import ArgumentParser

    path = "/Users/pgrynfelder/Library/CloudStorage/GoogleDrive-wadh6460@ox.ac.uk/My Drive/results"

    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--Np", type=int, required=True)
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--ls", type=float, default=None)
    parser.add_argument("--var", type=float, default=None)
    parser.add_argument("--num_ctx", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--noise", type=float, default=None)
    args = parser.parse_args()

    apply, config = load_model(path)
    run(apply=apply, config=config, **vars(args))
