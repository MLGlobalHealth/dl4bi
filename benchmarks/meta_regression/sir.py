#!/usr/bin/env python3
import math
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib as mpl
import optax
import wandb
from jax import random
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid

from dl4bi.meta_regression.train_utils import (
    Callback,
    cfg_to_run_name,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    log_img_plots,
    save_ckpt,
    select_steps,
    train,
)
from dl4bi.meta_regression.transform import pointwise_multinomial


# TODO(danj): configure plotting cmap
@hydra.main("configs/sir", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    dataloader = build_dataloader(cfg.data, cfg.sim)
    lr_schedule = cosine_annealing_lr(
        cfg.train_num_steps,
        cfg.lr_peak,
        cfg.lr_pct_warmup,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.clip_max_norm),
        optax.yogi(lr_schedule),
    )
    model = instantiate(cfg.model)
    train_step, valid_step = select_steps(model, is_categorical=True)
    cmap = mpl.colormaps.get_cmap("Spectral_r")
    cmap.set_bad("grey")
    dims = [dim.num for dim in cfg.data.s]
    clbk = partial(
        log_img_plots,
        shape=(*dims, 3),
        num_plots=16,
        cmap=cmap,
        transform_model_output=pointwise_multinomial,
    )
    state = train(
        rng_train,
        model,
        optimizer,
        train_step,
        valid_step,
        dataloader,
        dataloader,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[Callback(clbk, cfg.plot_interval)],
    )
    metrics = evaluate(
        rng_test,
        state,
        valid_step,
        dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloader(data: DictConfig, priors: DictConfig):
    """A 2D Lattice SIR dataloader."""
    si = instantiate(priors)
    dims, D_s = [dim.num for dim in data.s], len(data.s)
    Lc_min, Lc_max, Lt = data.num_ctx.min, data.num_ctx.max, data.num_test
    L, B, N = math.prod(dims), data.batch_size, data.num_steps
    s_grid = build_grid(data.s).reshape(-1, D_s)  # flatten spatial dims
    s = jnp.repeat(s_grid[None, ...], B, axis=0)
    valid_lens_test = jnp.repeat(Lt, B)

    def dataloader(rng: jax.Array):
        while True:
            rng_si, rng_eps, rng_valid, rng_permute, rng = random.split(rng, 5)
            steps, *_ = si.simulate(rng_si, dims, N)  # f: [N, *dims]
            i, step = -1, steps[-1]
            while (step == 1.0).all():
                i -= 1
                step = steps[i]
            steps = steps[:i]  # remove steps that consist of all infected
            steps = jax.nn.one_hot(steps + 1, 3)  # convert [-1, 0, 1] -> one hot
            N_sim = steps.shape[0]
            permute_idx = random.choice(rng_permute, N_sim, (N_sim,), replace=False)
            steps = steps[permute_idx]
            rng_permute, rng = random.split(rng)
            for i in range(N_sim // B):
                steps_i = steps[i * B : (i + 1) * B].reshape(B, L, 3)
                permute_idx = random.choice(rng_permute, L, (L,), replace=False)
                inv_permute_idx = jnp.argsort(permute_idx)
                valid_lens_ctx = random.randint(rng_valid, (B,), Lc_min, Lc_max)
                s_perm = s[:, permute_idx, :]
                f_perm = steps_i[:, permute_idx, :]
                yield (
                    s_perm[:, :Lc_max, :],
                    f_perm[:, :Lc_max, :],
                    valid_lens_ctx,
                    s_perm[:, Lc_max : Lc_max + Lt, :],
                    f_perm[:, Lc_max : Lc_max + Lt, :],
                    valid_lens_test,
                    s,  # add full originals for use in callbacks, e.g. log_plots
                    steps_i,
                    inv_permute_idx,
                )

    return dataloader


if __name__ == "__main__":
    main()
