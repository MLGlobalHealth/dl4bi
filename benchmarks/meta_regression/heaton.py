#!/usr/bin/env python3
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import pandas as pd
from jax import random
from omegaconf import DictConfig, OmegaConf

import wandb
from dl4bi.meta_regression.train_utils import (
    Callback,
    TrainState,
    cfg_to_run_name,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    log_img_plots,
    save_ckpt,
    train,
)

# NOTE: uncomment to speed up on NVIDIA GPUs
# https://jax.readthedocs.io/en/latest/gpu_performance_tips.html#code-generation-flags
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_enable_triton_softmax_fusion=true '
#     '--xla_gpu_triton_gemm_any=True '
#     '--xla_gpu_enable_async_collectives=true '
#     '--xla_gpu_enable_latency_hiding_scheduler=true '
#     '--xla_gpu_enable_highest_priority_async_stream=true '
# )


@hydra.main("configs/heaton", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=cfg.get("name", run_name),
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    dataloader = build_dataloader(
        cfg.data.path,
        cfg.data.batch_size,
        cfg.data.num_ctx.min,
        cfg.data.num_ctx.max,
        cfg.data.num_test.max,
    )
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
    state = train(
        rng_train,
        model,
        optimizer,
        dataloader,
        dataloader,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[
            Callback(partial(log_img_plots, shape=(300, 500, 1)), cfg.plot_interval)
        ],
    )
    loss = evaluate(rng_test, state, dataloader, cfg.valid_num_steps)
    wandb.log({"test_loss": loss})
    path = Path(f"results/heaton/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloader(
    path: Path,
    batch_size: int = 16,
    num_ctx_min: int = 200,
    num_ctx_max: int = 500,
    num_test_max: int = 1000,
):
    path = Path("cache/heaton/sim.csv")
    df = pd.read_csv(path)
    df = preprocess(df)
    s_ctx, f_ctx, s_test = ctx_test_split(df)
    B, L_ctx = batch_size, s_ctx.shape[0]
    valid_lens_test = jnp.repeat(num_test_max, B)

    # For training and validation, we use the context points (s_ctx, f_ctx)
    # as a dataset; only when we run the final test do we input all (s_ctx, f_ctx)
    # and try to predict f_test at s_test locations.
    def dataloader(rng: jax.Array):
        while True:
            rng_permute, rng_valid, rng = random.split(rng, 3)
            s_ctxs, f_ctxs, s_tests, f_tests = [], [], [], []
            for _ in range(B):  # TODO(danj): speed up?
                rng_i, rng_permute = random.split(rng_permute)
                permute_idx = random.choice(rng_i, L_ctx, (L_ctx,), replace=False)
                s_ctxs += [s_ctx[permute_idx, :][:num_ctx_max, :]]
                f_ctxs += [f_ctx[permute_idx, :][:num_ctx_max, :]]
                s_tests += [s_ctx[permute_idx, :][:num_test_max, :]]
                f_tests += [f_ctx[permute_idx, :][:num_test_max, :]]
            valid_lens_ctx = random.randint(rng_valid, (B,), num_ctx_min, num_ctx_max)
            yield (
                jnp.stack(s_ctxs),
                jnp.stack(f_ctxs),
                valid_lens_ctx,
                jnp.stack(s_tests),
                jnp.stack(f_tests),
                valid_lens_test,
                s_ctx,  # return full originals for log_plot callback
                f_ctx,
                s_test,
            )

    return dataloader


def preprocess(df: pd.DataFrame):
    df.Lon -= df.Lon.mean()
    df.Lat -= df.Lat.mean()
    df.Temp = (df.Temp - df.Temp.mean()) / df.Temp.std()
    return df


def ctx_test_split(df: pd.DataFrame):
    ctx_idx = df.Temp.notna().values
    ctx, test = df[ctx_idx].values, df[~ctx_idx].values
    s_ctx, f_ctx = ctx[:, :-1], ctx[:, [-1]]
    s_test = test[:, :-1]  # f_test is all nans
    return s_ctx, f_ctx, s_test


def log_plot(step: int, rng_step: int, state: TrainState, batch: tuple):
    *_, s_ctx, f_ctx, s_test = batch
    rng_dropout, rng_extra = random.split(rng_step)
    f_mu, f_std, *_ = state.apply_fn(
        {"params": state.params, **state.kwargs},
        s_ctx,
        f_ctx,
        s_test,
        valid_lens_ctx=None,
        valid_lens_test=None,
        rngs={"dropout": rng_dropout, "extra": rng_extra},
    )
    s = jnp.vstack([s_ctx, s_test])
    f = jnp.vstack([f_ctx, f_mu])
    df = pd.DataFrame(jnp.hstack([s, f]), columns=["Lon", "Lat", "Temp"])


def plot_sat(df: pd.DataFrame, path: Path):
    df = df.sort_values(["Lat", "Lon"], ascending=[False, True])
    plt.imshow(df.Temp.values.reshape(300, 500), cmap="inferno", interpolation="none")
    plt.savefig(path, dpi=600)


def plot_img(
    id: int,
    shape: tuple[int, int, int],
    f_ctx: jax.Array,  # [L_ctx, 1]
    f_mu: jax.Array,  # [L, 1]
    f_test: jax.Array,  # [L, 1]
    inv_permute_idx: jax.Array,  # [L]
):
    """Plots a triptych of [task, pred, truth]."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(task)
    axs[0].set_title("Task")
    axs[1].imshow(task_pred)
    axs[1].set_title("Predicted")
    axs[2].imshow(task_true)
    axs[2].set_title("Ground Truth")
    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    main()
