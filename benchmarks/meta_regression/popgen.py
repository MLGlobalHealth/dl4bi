#!/usr/bin/env python3
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from hydra.core.hydra_config import HydraConfig
from jax import random
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid

from dsp.meta_regression.train_utils import (
    Callback,
    cosine_annealing_lr,
    evaluate,
    log_img_plots,
    save_ckpt,
    train,
)


@hydra.main("configs/popgen", version_base=None)
def main(cfg: DictConfig):
    d = HydraConfig.get().runtime.choices
    model_cfg_name = d["model"]
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if "wandb" in cfg else "disabled",
        name=cfg.get("name", model_cfg_name),
        project="SPTx - Population Genetics",
    )
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    dataloader = build_dataloader()
    train_num_steps, valid_num_steps, test_num_steps = 100000, 5000, 5000
    valid_interval, plot_interval = 25000, 50000
    lr_peak, lr_pct_warmup = 5e-4, 0.3
    lr_schedule = cosine_annealing_lr(train_num_steps, lr_peak, lr_pct_warmup)
    optimizer = optax.yogi(lr_schedule)
    state = train(
        rng_train,
        cfg.model,
        optimizer,
        dataloader,
        dataloader,
        train_num_steps,
        valid_num_steps,
        valid_interval,
        callbacks=[Callback(partial(log_img_plots, shape=(32, 32, 1)), plot_interval)],
    )
    path = Path(f"results/popgen/{model_cfg_name}-seed-{cfg.seed}")
    path.parent.mkdir(parents=True, exist_ok=True)
    loss = evaluate(
        rng_test,
        state,
        dataloader,
        test_num_steps,
        path.with_suffix(".pkl"),
    )
    wandb.log({"test_loss": loss})
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloader(
    batch_size: int = 32,
    num_ctx_min: int = 100,
    num_ctx_max: int = 300,
    num_test_max: int = 400,
):
    B, L = batch_size, 32 * 32
    data = np.load("cache/popgen/n1000_mu_1e-5_m_5e-3.npy", allow_pickle=True).item()
    train_ds = data["f_test"]
    s_test = build_grid([dict(start=-1.0, stop=1.0, num=32)] * 2).reshape(L, 2)
    s_test = jnp.repeat(s_test[None, ...], B, axis=0)  # [L, 2] -> [B, L, 2]
    valid_lens_test = jnp.repeat(num_test_max, B)

    def build_dataloader(dataset):
        N = dataset.shape[0]

        def dataloader(rng: jax.Array):
            while True:
                rng_batch, rng_permute, rng_valid, rng = random.split(rng, 4)
                batch_idx = random.choice(rng_batch, N, (B,), replace=False)
                permute_idx = random.choice(rng_permute, L, (L,), replace=False)
                f_test = dataset[batch_idx]
                f_test = f_test.reshape(B, -1, 1)  # [B, H, W, 1] -> [B, L, 1]
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

    return build_dataloader(train_ds)


if __name__ == "__main__":
    main()
