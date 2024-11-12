#!/usr/bin/env python3
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
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
    train,
)


@hydra.main("configs/cifar_10", config_name="default", version_base=None)
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
    train_dataloader, valid_dataloader, test_dataloader = build_dataloaders()
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
        train_dataloader,
        valid_dataloader,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[
            Callback(partial(log_img_plots, shape=(32, 32, 3)), cfg.plot_interval)
        ],
    )
    metrics = evaluate(rng_test, state, test_dataloader, cfg.valid_num_steps)
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloaders(
    batch_size: int = 32,
    num_ctx_min: int = 3,
    num_ctx_max: int = 100,
    num_test_max: int = 200,
):
    B, L = batch_size, 32 * 32
    prepare_data()
    train_ds = 2 * (np.load("cache/cifar_10/train.npy", mmap_mode="r") / 255.0 - 0.5)
    valid_ds = 2 * (np.load("cache/cifar_10/valid.npy", mmap_mode="r") / 255.0 - 0.5)
    test_ds = 2 * (np.load("cache/cifar_10/test.npy", mmap_mode="r") / 255.0 - 0.5)
    s_test = build_grid([dict(start=-1.0, stop=1.0, num=32)] * 2).reshape(L, 2)
    s_test = jnp.repeat(s_test[None, ...], B, axis=0)  # [L, 2] -> [B, L, 2]
    valid_lens_test = jnp.repeat(num_test_max, B)

    def build_dataloader(dataset):
        N = dataset.shape[0]

        def dataloader(rng: jax.Array):
            while True:
                rng_batch, rng_permute, rng_valid, rng = jax.random.split(rng, 4)
                batch_idx = jax.random.choice(rng_batch, N, (B,), replace=False)
                permute_idx = jax.random.choice(rng_permute, L, (L,), replace=False)
                f_test = dataset[batch_idx]
                f_test = f_test.reshape(B, -1, 3)  # [B, H, W, 3] -> [B, L, 3]
                inv_permute_idx = jnp.argsort(permute_idx)

                s_test_permuted = s_test[:, permute_idx, :]
                f_test_permuted = f_test[:, permute_idx, :]
                s_test_permuted = s_test_permuted[:, :num_test_max, :]
                f_test_permuted = f_test_permuted[:, :num_test_max, :]
                valid_lens_ctx = jax.random.randint(
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
                    s_test,  # full originals
                    f_test,
                    inv_permute_idx,
                )

        return dataloader

    return (
        build_dataloader(train_ds),
        build_dataloader(valid_ds),
        build_dataloader(test_ds),
    )


def prepare_data():
    """Prepares and caches the CIFAR-10 dataset with standard splits."""
    cache_path = Path("cache/cifar_10")
    cache_path.mkdir(parents=True, exist_ok=True)
    train_path = cache_path / "train.npy"
    valid_path = cache_path / "valid.npy"
    test_path = cache_path / "test.npy"

    if train_path.exists() and valid_path.exists() and test_path.exists():
        return

    dataset, info = tfds.load(
        "cifar10",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
        data_dir=cache_path,
    )
    train_dataset, test_dataset = dataset[0], dataset[1]
    num_examples = info.splits["train"].num_examples
    train_dataset = train_dataset.shuffle(num_examples, seed=42)
    train_split = int(info.splits["train"].num_examples * 0.9)
    train_data, valid_data = (
        tf.data.Dataset.take(train_dataset, train_split),
        tf.data.Dataset.skip(train_dataset, train_split),
    )
    train_data = dataset_to_numpy(train_data)
    valid_data = dataset_to_numpy(valid_data)
    test_data = dataset_to_numpy(test_dataset)
    np.save(train_path, train_data)
    np.save(valid_path, valid_data)
    np.save(test_path, test_data)


def dataset_to_numpy(dataset):
    """Converts a TFDS dataset into a numpy array."""
    images = []
    for image, _ in tfds.as_numpy(dataset):
        images.append(image)
    return np.array(images, dtype=np.float32)


if __name__ == "__main__":
    main()
