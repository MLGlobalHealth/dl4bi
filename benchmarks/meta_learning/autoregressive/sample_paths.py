from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
from dataloader import build_gp_dataloader
from jax import random
from omegaconf import OmegaConf

from dl4bi.meta_learning.autoregressive import AutoregressiveSampler
from dl4bi.meta_learning.train_utils import load_ckpt

device = jax.devices()[0]


def run(
    *,
    sampler: AutoregressiveSampler,
    config: OmegaConf,
    seed: int,
    job_name: str,
    output_dir: Path,
    num_paths: int,  # number of paths to sample autoregressively, will be rounded up to be a multiple of batch size
    batch_size: int,  # how to batch the autoregressive sampling. this parameter only affects runtime efficiency
    ls: float | None = None,  # override the ls in the config
    var: float | None = None,  # override the var in the config
    num_ctx: int | None = None,  # override number of context points to use
    obs_noise: float | None = None,  # override the noise in the config
    s_test: list[float] | None = None,  # override s_test
):
    rng = random.key(seed)

    output_dir = output_dir / job_name
    output_dir.mkdir(parents=True)
    print("Results dir:", output_dir)
    jnp.save(output_dir / "seed.npy", seed)

    rng, rng_dataloader = jax.random.split(rng)

    config = config.copy()
    config.data.batch_size = 1
    if ls is not None:
        config.kernel.kwargs.ls.kwargs = {"dist": "fixed", "kwargs": {"value": ls}}
    if var is not None:
        config.kernel.kwargs.var.kwargs = {"dist": "fixed", "kwargs": {"value": var}}

    if num_ctx is not None:
        config.data.num_ctx = {"min": num_ctx, "max": num_ctx}
    if obs_noise is not None:
        config.data.obs_noise = obs_noise

    OmegaConf.save(config, output_dir / "config.yaml")
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
        [f_obs],
        _valid_lens_test,  # unused
        [var],
        [ls],
        _period,  # unused
    ) = next(dataloader)
    print(f"var {var}, ls {ls}")
    jnp.save(output_dir / "var.npy", var)
    jnp.save(output_dir / "ls.npy", ls)

    f_ctx = f_ctx[:valid_lens_ctx]
    s_ctx = s_ctx[:valid_lens_ctx]

    if s_test is None:
        # Excluding the context points from test set for numerical stability of later analysis.
        num_ctx = config.data.num_ctx.max
        s_test = s[num_ctx:]
    else:
        s_test = jnp.array(s_test, dtype=jnp.float32).reshape(len(s_test), 1)

    # Save everything that can be needed for later analysis
    jnp.save(output_dir / "s_ctx.npy", s_ctx)
    jnp.save(output_dir / "f_ctx.npy", f_ctx)
    jnp.save(output_dir / "s_test.npy", s_test)
    jnp.save(output_dir / "s.npy", s)
    jnp.save(output_dir / "f.npy", f)

    # Regular (diagonal) TNP-KR
    [diagonal_mu], [diagonal_sd] = sampler.model(
        s_ctx[None],
        f_ctx[None],
        s_test[None],
    )  # note need to expand dims
    diagonal_var = diagonal_sd**2
    jnp.save(output_dir / "diagonal_mu", diagonal_mu)
    jnp.save(output_dir / "diagonal_var", diagonal_var)

    strategies = ["ltr", "random", "furthest", "closest"]
    for strategy in strategies:
        start = datetime.now()
        paths = sampler.sample_multiple_paths(
            rng,
            s_ctx,
            f_ctx,
            s_test,
            batch_size,
            num_paths,
            strategy,
        )
        end = datetime.now()
        jnp.save(output_dir / f"{strategy}_paths.npy", paths)
        print(f"{strategy} took {end - start}")


if __name__ == "__main__":
    from argparse import ONE_OR_MORE, ArgumentParser

    # TODO @pgrynfelder: can we somehow integrate this with hydra?
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("results"))
    parser.add_argument("-j", "--job-name", type=str, default=str(datetime.now()))
    parser.add_argument("-N", "--num-paths", type=int, default=128)
    parser.add_argument("-B", "--batch-size", type=int, default=16)
    parser.add_argument("-l", "--ls", type=float, default=None)
    parser.add_argument("-v", "--var", type=float, default=None)
    parser.add_argument("-C", "--num-ctx", type=int, default=None)
    parser.add_argument("--obs-noise", type=float, default=None)
    parser.add_argument("--s-test", type=float, nargs=ONE_OR_MORE, default=None)
    args = parser.parse_args()

    state, config = load_ckpt(args.path)
    sampler = AutoregressiveSampler.from_state(state)

    print(args)

    run(
        sampler=sampler,
        config=config,
        seed=args.seed,
        job_name=args.job_name,
        output_dir=args.output_dir,
        num_paths=args.num_paths,
        batch_size=args.batch_size,
        ls=args.ls,
        var=args.var,
        num_ctx=args.num_ctx,
        obs_noise=args.obs_noise,
        s_test=args.s_test,
    )
