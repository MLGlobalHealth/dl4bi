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
from hydra.utils import instantiate
from jax import random
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from sps.utils import build_grid
from tqdm import tqdm

from dl4bi.core.train import (
    Callback,
    cosine_annealing_lr,
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.spatial import SpatialData
from dl4bi.meta_learning.utils import (
    cfg_to_run_name,
    wandb_2d_img_callback,
)


@hydra.main("configs/celeba", config_name="default", version_base=None)
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
    output_fn = model.output_fn
    model = model.copy(output_fn=lambda x: output_fn(x, min_std=0.05))
    state = train(
        rng_train,
        model,
        optimizer,
        model.train_step,
        model.valid_step,
        train_dataloader,
        valid_dataloader,
        clbk_dataloader,  # callback_dataloader
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[Callback(wandb_2d_img_callback, cfg.plot_interval)],
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
    prepare_data()
    B, L = batch_size, 32 * 32
    # load & convert from [0, 255] -> [-0.5, 0.5] -> [-1, 1]
    train_ds = 2 * (np.load("cache/celeba/train.npy", mmap_mode="r") / 255.0 - 0.5)
    valid_ds = 2 * (np.load("cache/celeba/valid.npy", mmap_mode="r") / 255.0 - 0.5)
    test_ds = 2 * (np.load("cache/celeba/test.npy", mmap_mode="r") / 255.0 - 0.5)
    s = build_grid([dict(start=-2.0, stop=2.0, num=32)] * 2).reshape(L, 2)
    s = jnp.repeat(s[None], B, axis=0)  # [B, S, S, 2]

    def build_dataloader(dataset, is_callback: bool = False):
        N = dataset.shape[0]

        def dataloader(rng: jax.Array):
            while True:
                rng_i, rng_b, rng = random.split(rng, 3)
                batch_idx = random.choice(rng_i, N, (B,), replace=False)
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
    if train_path.exists() and valid_path.exists() and test_path.exists():
        # skip if training data directly copied without extraction from zip
        return

    if not imgs_zip_path.exists():
        request.urlretrieve(imgs_url, imgs_zip_path)

    if not part_path.exists():
        request.urlretrieve(part_url, part_path)

    if not imgs_path.exists():
        try:
            shutil.unpack_archive(imgs_zip_path, cache_path)
        except shutil.ReadError:
            msg = "\n\nFailed to unpack imgs_align_celeba.zip."
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
