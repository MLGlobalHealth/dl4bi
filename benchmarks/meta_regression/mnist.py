#!/usr/bin/env python3
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from hydra.core.hydra_config import HydraConfig
from jax import random
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid

from dsp.meta_regression.train_utils import (
    Callback,
    TrainState,
    save_ckpt,
    train,
    validate,
)


@hydra.main("configs/mnist", version_base=None)
def main(cfg: DictConfig):
    d = HydraConfig.get().runtime.choices
    model_name = d["model"]
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if "wandb" in cfg else "disabled",
        name=cfg.get("name", f"mnist - {model_name} - seed {cfg.seed}"),
        project="SPTx - MNIST",
    )
    rng = random.key(cfg.seed)
    rng_train, rng_valid = random.split(rng)
    train_dataloader, valid_dataloader = build_dataloaders()
    train_num_steps, valid_num_steps = 100000, None  # exhaust valid dataloader
    valid_interval, plot_interval = 25000, 25000
    optimizer = optax.yogi(1e-4)
    state = train(
        rng_train,
        cfg.model,
        optimizer,
        train_dataloader,
        valid_dataloader,
        train_num_steps,
        valid_num_steps,
        valid_interval,
        callbacks=[Callback(log_plots, plot_interval)],
    )
    path = Path(f"results/mnist/{model_name}-seed-{cfg.seed}")
    path.parent.mkdir(parents=True, exist_ok=True)
    loss = validate(
        rng_valid,
        state,
        valid_dataloader,
        valid_num_steps,
        path.with_suffix(".pkl"),
    )
    wandb.log({"test_loss", loss})
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloaders(
    batch_size: int = 16,
    num_ctx_min: int = 3,
    num_ctx_max: int = 200,
    num_test_max: int = 200,
):
    B, L = batch_size, 28 * 28
    normalize = lambda sample: tf.cast(sample["image"], tf.float32) / 255.0
    train_ds = tfds.load("mnist", split="train").map(normalize)
    valid_ds = tfds.load("mnist", split="test").map(normalize)
    train_ds = train_ds.repeat().batch(batch_size).prefetch(1)
    valid_ds = valid_ds.batch(batch_size).prefetch(1)
    s_test = build_grid([dict(start=-1.0, stop=1.0, num=28)] * 2).reshape(L, 2)
    s_test = jnp.repeat(s_test[None, ...], B, axis=0)  # [L, 2] -> [B, L, 2]
    valid_lens_test = jnp.repeat(num_test_max, B)  # similar to ANP, Appendix D

    def build_dataloader(dataset):
        def dataloader(rng: jax.Array):
            for f_test in dataset.as_numpy_iterator():
                rng_permute, rng_valid, rng = random.split(rng, 3)
                f_test = f_test.reshape(B, -1, 1)  # [B, H, W, 1] -> [B, L, 1]
                permute_idx = random.choice(rng_permute, L, (L,), replace=False)
                inv_permute_idx = jnp.argsort(permute_idx)
                # permute the order and select the first valid_lens_ctx for context
                s_test_permuted = s_test[:, permute_idx, :]
                f_test_permuted = f_test[:, permute_idx, :]
                s_test_permuted = s_test_permuted[:, :num_test_max, :]
                f_test_permuted = f_test_permuted[:, :num_test_max, :]
                valid_lens_ctx = random.randint(
                    rng_valid,
                    (B,),
                    num_ctx_min,
                    num_ctx_max,
                )
                yield (
                    s_test_permuted,  # s_ctx (permuted)
                    f_test_permuted,  # f_ctx (permuted)
                    valid_lens_ctx,  # only the first valid lens are used/observed
                    s_test_permuted,  # s_test (permuted)
                    f_test_permuted,  # f_test (permuted)
                    valid_lens_test,
                    s_test,  # add full originals for use in callbacks, e.g. log_plots
                    f_test,
                    inv_permute_idx,
                )

        return dataloader

    return build_dataloader(train_ds), build_dataloader(valid_ds)


def log_plots(
    step: int,
    rng_step: int,
    state: TrainState,
    batch: tuple,
    num_plots: int = 16,
):
    """Logs `num_plots` from the given batch."""
    rng_dropout, rng_extra = random.split(rng_step)
    (
        s_ctx,
        f_ctx,
        valid_lens_ctx,
        s_test,
        f_test,
        valid_lens_test,
        s_test_full,
        f_test_full,
        inv_permute_idx,
    ) = batch
    f_mu, f_std, *_ = state.apply_fn(
        {"params": state.params, **state.kwargs},
        s_ctx,
        f_ctx,
        s_test_full,
        valid_lens_ctx,
        valid_lens_test=None,
        rngs={"dropout": rng_dropout, "extra": rng_extra},
    )
    paths = []
    for i in range(num_plots):
        v = valid_lens_ctx[i]
        f_ctx_i = f_ctx[i, :v, :]
        f_mu_i = f_mu[i]
        f_test_full_i = f_test_full[i]
        if f_mu.shape != f_test.shape:  # bootstrapped
            K = f_mu.shape[0] // f_test.shape[0]
            s = i * K
            f_mu_i = f_mu[s : s + K].mean(axis=0)  # TODO(danj): legitimate?
        path = plot_example(i, f_ctx_i, f_mu_i, f_test_full_i, inv_permute_idx)
        paths += [path]
    wandb.log({f"Step {step}": [wandb.Image(p) for p in paths]})


def plot_example(
    id: int,
    f_ctx: jax.Array,  # [L_ctx, 1]
    f_mu: jax.Array,  # [L, 1]
    f_test: jax.Array,  # [L, 1]
    inv_permute_idx: jax.Array,  # [L]
):
    """Plots a triptych of [task, pred, truth]."""
    task = f_ctx_to_task(f_ctx, inv_permute_idx)
    task_pred = f_to_task(f_mu)
    task_true = f_to_task(f_test)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(task)
    axs[0].set_title("Task")
    axs[1].imshow(task_pred)
    axs[1].set_title("Predicted")
    axs[2].imshow(task_true)
    axs[2].set_title("Ground Truth")
    path = f"/tmp/mnist_sample_{id}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.clf()
    plt.close(fig)
    return path


def f_ctx_to_task(f_ctx: jax.Array, inv_permute_idx: jax.Array):
    L, L_ctx = 28 * 28, f_ctx.shape[0]
    task = jnp.pad(f_ctx, ((0, L - L_ctx), (0, 0)))  # [L_ctx, 1] -> [L, 1]
    task = jnp.repeat(task, 3, axis=-1)  # [L, 1] -> [L, 3]
    task = task.at[L_ctx:, 2].set(1.0)  # set non-context points to blue
    task = task[inv_permute_idx, :]  # permute back to original ordering
    return task.reshape(28, 28, 3)  # reshape to [28, 28, 3] image


def f_to_task(f: jax.Array):
    task = f.reshape(28, 28, 1)  # [H, W, 1]
    task = jnp.repeat(task, 3, axis=-1)  # [H, W, 3]
    return jnp.clip(task, 0, 1)  # to avoid matplotlib warnings


if __name__ == "__main__":
    main()
