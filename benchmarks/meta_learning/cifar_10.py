#!/usr/bin/env python3
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from hydra.utils import instantiate
from jax import random
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid

from dl4bi.core.train import (
    Callback,
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.spatial import SpatialData
from dl4bi.meta_learning.utils import (
    cfg_to_run_name,
    regression_to_rgb,
    wandb_2d_img_callback,
)

# NOTE: this requires `tensorflow` and `tensorflow-datasets` packages


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
    train_dataloader, valid_dataloader, test_dataloader, clbk_dataloader = (
        build_dataloaders()
    )
    optimizer = instantiate(cfg.optimizer)
    model = instantiate(cfg.model)
    output_fn = model.output_fn
    model = model.copy(output_fn=lambda x: output_fn(x, min_std=0.05))
    clbk = partial(
        wandb_2d_img_callback,
        remap_colors=regression_to_rgb,
        filename_prefix="cifar_10",
    )
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
    )
    metrics = evaluate(
        rng_test,
        state,
        model.valid_step,
        test_dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloaders(
    batch_size: int = 16,
    num_ctx_min: int = 16,
    num_ctx_max: int = 128,
    num_test: int = 256,
):
    B = batch_size
    prepare_data()
    train_ds = 2 * (np.load("cache/cifar_10/train.npy", mmap_mode="r") / 255.0 - 0.5)
    valid_ds = 2 * (np.load("cache/cifar_10/valid.npy", mmap_mode="r") / 255.0 - 0.5)
    test_ds = 2 * (np.load("cache/cifar_10/test.npy", mmap_mode="r") / 255.0 - 0.5)
    s = build_grid([dict(start=-2.0, stop=2.0, num=32)] * 2)
    s = jnp.repeat(s[None, ...], B, axis=0)

    def build_dataloader(dataset, is_callback: bool = False):
        N = dataset.shape[0]

        def dataloader(rng: jax.Array):
            while True:
                rng_i, rng_b, rng = random.split(rng, 3)
                batch_idx = jax.random.choice(rng_i, N, (B,), replace=False)
                f = dataset[batch_idx]
                d = SpatialData(x=None, s=s, f=f)
                yield d.batch(
                    rng_b,
                    num_ctx_min,
                    num_ctx_max,
                    32 * 32 if is_callback else num_test,
                    test_includes_ctx=True,
                )

        return dataloader

    return (
        build_dataloader(train_ds),
        build_dataloader(valid_ds),
        build_dataloader(test_ds),
        build_dataloader(valid_ds, True),
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
