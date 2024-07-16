#!/usr/bin/env python3
import shutil
import sys
from functools import partial
from pathlib import Path
from urllib import request

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import wandb
from hydra.core.hydra_config import HydraConfig
from jax import random
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from sps.utils import build_grid
from tqdm import tqdm

from dsp.meta_regression.train_utils import (
    Callback,
    cosine_annealing_lr,
    evaluate,
    log_img_plots,
    save_ckpt,
    train,
)


@hydra.main("configs/celeba", version_base=None)
def main(cfg: DictConfig):
    d = HydraConfig.get().runtime.choices
    model_cfg_name = d["model"]
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if "wandb" in cfg else "disabled",
        name=cfg.get("name", model_cfg_name),
        project="SPTx - CelebA",
    )
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    train_dataloader, valid_dataloader, test_dataloader = build_dataloaders()
    # TODO(danj): fix
    # train_num_steps, valid_num_steps, test_num_steps = 100000, None, None
    train_num_steps, valid_num_steps, test_num_steps = 100000, 5000, 5000
    valid_interval, plot_interval = 25000, 50000
    lr_peak, lr_pct_warmup = 5e-4, 0.3
    lr_schedule = cosine_annealing_lr(train_num_steps, lr_peak, lr_pct_warmup)
    optimizer = optax.yogi(lr_schedule)
    state = train(
        rng_train,
        cfg.model,
        optimizer,
        train_dataloader,
        valid_dataloader,
        train_num_steps,
        valid_num_steps,
        valid_interval,
        callbacks=[Callback(partial(log_img_plots, shape=(32, 32, 3)), plot_interval)],
    )
    path = Path(f"results/celeba/{model_cfg_name}-seed-{cfg.seed}")
    path.parent.mkdir(parents=True, exist_ok=True)
    loss = evaluate(
        rng_test,
        state,
        test_dataloader,
        test_num_steps,
        path.with_suffix(".pkl"),
    )
    wandb.log({"test_loss": loss})
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloaders(
    batch_size: int = 16,
    num_ctx_min: int = 3,
    num_ctx_max: int = 200,
    num_test_max: int = 200,
):
    prepare_data()
    B, L = batch_size, 32 * 32
    train_ds = np.load("cache/celeba/train.npy")
    valid_ds = np.load("cache/celeba/valid.npy")
    test_ds = np.load("cache/celeba/test.npy")
    s_test = build_grid([dict(start=-1.0, stop=1.0, num=32)] * 2).reshape(L, 2)
    s_test = jnp.repeat(s_test[None, ...], B, axis=0)  # [L, 2] -> [B, L, 2]
    valid_lens_test = jnp.repeat(num_test_max, B)  # similar to ANP, Appendix D

    def build_dataloader(dataset):
        N = dataset.shape[0]

        # TODO(danj): update for valid/test
        def dataloader(rng: jax.Array):
            while True:
                rng_batch, rng_permute, rng_valid, rng = random.split(rng, 4)
                batch_idx = random.choice(rng_batch, N, (B,), replace=False)
                permute_idx = random.choice(rng_permute, L, (L,), replace=False)
                f_test = dataset[batch_idx]
                f_test = f_test.reshape(B, -1, 3)  # [B, H, W, 3] -> [B, L, 3]
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

    return (
        build_dataloader(train_ds),
        build_dataloader(valid_ds),
        build_dataloader(test_ds),
    )


def prepare_data():
    """Prepares a 32x32 CelebA dataset."""
    # NOTE: as of 2024-07-16, Tensorflow Dataset's celeb_a dataset has an
    # invalid checksum and fails to download and so we do it manually here
    # Issue: https://github.com/tensorflow/datasets/issues/1482
    cache_path = Path("cache/celeba")
    cache_path.mkdir(parents=True, exist_ok=True)
    imgs_zip_path = cache_path / "img_align_celeba.zip"
    imgs_path = cache_path / "img_align_celeba"
    part_path = cache_path / "list_eval_partition.txt"
    train_path = cache_path / "train.npy"
    valid_path = cache_path / "valid.npy"
    test_path = cache_path / "test.npy"

    imgs_url, part_url = (
        "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM",
        "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pY0NSMzRuSXJEVkk",
    )
    if not imgs_zip_path.exists():
        request.urlretrieve(imgs_url, imgs_zip_path)

    if not part_path.exists():
        request.urlretrieve(part_url, part_path)

    if not imgs_path.exists():
        try:
            shutil.unpack_archive(imgs_zip_path, cache_path)
        except shutil.ReadError:
            msg = "Failed to unpack imgs_align_celeba.zip."
            msg += " This likely means the download failed."
            msg += " Please see the README for instructions"
            msg += " on downloading the dataset manually."
            sys.exit(msg)

    if any([not p.exists() for p in [train_path, valid_path, test_path]]):
        df = pd.read_csv(part_path, sep=" ", header=None)
        df.columns = ["path", "part"]

    if not train_path.exists():
        imgs = preprocess(df, "train", imgs_path)
        np.save(train_path, imgs)
    if not valid_path.exists():
        imgs = preprocess(df, "valid", imgs_path)
        np.save(valid_path, imgs)
    if not test_path.exists():
        imgs = preprocess(df, "test", imgs_path)
        np.save(test_path, imgs)


def preprocess(df: pd.DataFrame, split: str, imgs_dir: Path):
    imgs = []
    part = {"train": 0, "valid": 1, "test": 2}[split]
    desc = f"Preprocessing {split} split"
    for filename in tqdm(df[df.part == part].path.values, desc=desc):
        imgs += [crop_and_resize(imgs_dir / filename)]
    return imgs


def crop_and_resize(path: Path):
    """Crop and resize to 32x32."""
    img = Image.open(path)
    img = img.crop((15, 40, 15 + 148, 40 + 148))  # [148, 148, 3]
    img = img.resize((32, 32))
    return np.array(img, dtype=np.uint8)


if __name__ == "__main__":
    main()
