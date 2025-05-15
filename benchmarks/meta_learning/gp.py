#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import wandb
from hydra.utils import instantiate
from jax import jit, random
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid

from dl4bi.core.train import (
    Callback,
    TrainState,
    evaluate,
    load_ckpt,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.spatial import SpatialBatch, SpatialData
from dl4bi.meta_learning.utils import cfg_to_run_name  # , save_batches_for_tabpfn


@hydra.main("configs/gp", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project + cfg.project_suffix,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))
    kernel = cfg.kernel.kernel._target_.split(".")[-1]
    path = f"results/{cfg.project}/{cfg.data.name}/{kernel}/{cfg.seed}/{run_name}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    train_dataloader = valid_dataloader = build_dataloader(cfg.data, cfg.kernel)
    clbk_dataloader = build_dataloader(cfg.data, cfg.kernel, is_callback=True)
    optimizer = instantiate(cfg.optimizer)
    model = instantiate(cfg.model)
    if cfg.data.name == "2d":
        clbk = wandb_2d_plots
        clbk_dataloader = build_2d_grid_dataloader(cfg.data, cfg.kernel)
    if cfg.evaluate_only:
        # NOTE: pass override_cfg=cfg in case attributes have been updated,
        # e.g. ConvCNP needs updated bounds for its latent grid
        state, _ = load_ckpt(path.with_suffix(".ckpt"), cfg)
        metrics = evaluate(
            rng_test,
            state,
            model.valid_step,
            valid_dataloader,
            cfg.valid_num_steps,
            callbacks=[Callback(clbk, cfg.valid_num_steps)],
            callback_dataloader=clbk_dataloader,
        )
        wandb.log({f"Test {m}": v for m, v in metrics.items()})
        return
    clbk = wandb_1d_plots
    state = train(
        rng_train,
        model,
        optimizer,
        model.train_step,
        cfg.train_num_steps,
        train_dataloader,
        model.valid_step,
        cfg.valid_interval,
        cfg.valid_num_steps,
        valid_dataloader,
        callbacks=[Callback(clbk, cfg.plot_interval)],
        callback_dataloader=clbk_dataloader,
        return_state="last",
    )
    metrics = evaluate(
        rng_test,
        state,
        model.valid_step,
        valid_dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))
    eval_path = path.parent / f"eval_data.npy"
    # save_batches_for_tabpfn(rng_test, valid_dataloader, cfg.valid_num_steps, eval_path)


def build_dataloader(data: DictConfig, kernel: DictConfig, is_callback: bool = False):
    """Generates batches of GP samples."""
    gp = instantiate(kernel)
    B, L, D_s = data.batch_size, data.num_ctx.max + data.num_test, len(data.s)
    s_min = jnp.array([axis["start"] for axis in data.s])
    s_max = jnp.array([axis["stop"] for axis in data.s])
    batchify = jit(lambda x: jnp.repeat(x[None, ...], B, axis=0))
    to_extra = lambda d: {k: v.item() for k, v in d.items() if v is not None}

    def dataloader(rng: jax.Array):
        while True:
            rng_s, rng_gp, rng_b, rng = random.split(rng, 4)
            s = random.uniform(rng_s, (L, D_s), jnp.float32, s_min, s_max)
            f, var, ls, period, *_ = gp.simulate(rng_gp, s, B)
            s = batchify(s)
            d = SpatialData(x=None, s=s, f=f)
            b = d.batch(
                rng_b,
                data.num_ctx.min,
                data.num_ctx.max,
                num_test=data.num_test,
                test_includes_ctx=False,
                obs_noise=data.obs_noise,
            )
            if is_callback:
                yield b, to_extra({"var": var, "ls": ls, "period": period})
            else:
                yield b

    return dataloader


def build_2d_grid_dataloader(data: DictConfig, kernel: DictConfig):
    """A custom 2D GP dataloader in which generated context and test points
        reside only on the 2d grid.

    .. note::
        The dataloader used for training and testing uses context points
        on a continuous domain, while this only uses points on a grid for
        visualization purposes.
    """
    B = data.batch_size
    gp = instantiate(kernel)
    s_g = build_grid(data.s)
    s = jnp.repeat(s_g[None, ...], B, axis=0)
    to_extra = lambda d: {k: v.item() for k, v in d.items() if v is not None}

    def dataloader(rng: jax.Array):
        while True:
            rng_gp, rng_b, rng = random.split(rng, 3)
            f, var, ls, period, *_ = gp.simulate(rng_gp, s_g, B)
            f = f.reshape(*s.shape[:-1], f.shape[-1])
            d = SpatialData(x=None, s=s, f=f)
            b = d.batch(
                rng_b,
                data.num_ctx.min,
                data.num_ctx.max,
                num_test=s.shape[1] * s.shape[2],
                test_includes_ctx=True,
                obs_noise=data.obs_noise,
            )
            yield b, to_extra({"var": var, "ls": ls, "period": period})

    return dataloader


def wandb_1d_plots(
    step: int,
    rng_step: int,
    state: TrainState,
    batch: SpatialBatch,
    extra: dict,
    num_plots: int = 8,
):
    rng_dropout, rng_extra = random.split(rng_step)
    output = state.apply_fn(
        {"params": state.params, **state.kwargs},
        **batch,
        rngs={"dropout": rng_dropout, "extra": rng_extra},
    )
    if isinstance(output, tuple):
        output, _ = output  # throw away latent samples
    path = f"/tmp/gp_1d_{datetime.now().isoformat()}.png"
    subtitle = ", ".join([f"{k}: {v:.2f}" for k, v in extra.items()])
    fig = batch.plot_1d(output.mu, output.std, subtitle=subtitle, num_plots=num_plots)
    fig.savefig(path)
    plt.close(fig)
    wandb.log({f"Step {step}": wandb.Image(path)})


def wandb_2d_plots(
    step: int,
    rng_step: int,
    state: TrainState,
    batch: SpatialBatch,
    extra: dict,
    num_plots: int = 8,
):
    """Logs `num_plots` from the given batch for 2D GPs."""
    rng_dropout, rng_extra = random.split(rng_step)
    output = state.apply_fn(
        {"params": state.params, **state.kwargs},
        **batch,
        rngs={"dropout": rng_dropout, "extra": rng_extra},
    )
    if isinstance(output, tuple):
        output, _ = output  # throw away latent samples
    path = f"/tmp/gp_2d_{datetime.now().isoformat()}.png"
    cmap = mpl.colormaps.get_cmap("Spectral_r")
    cmap.set_bad("grey")
    subtitle = ", ".join([f"{k}: {v:.2f}" for k, v in extra.items()])
    fig = batch.plot_2d(
        output.mu,
        output.std,
        cmap=cmap,
        subtitle=subtitle,
        num_plots=num_plots,
    )
    fig.savefig(path)
    plt.close(fig)
    wandb.log({f"Step {step}": wandb.Image(path)})


if __name__ == "__main__":
    main()
