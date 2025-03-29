#!/usr/bin/env python3
from pathlib import Path
from typing import Callable

import flax.linen as nn
import hydra
import jax
import numpy as np
import wandb
from hydra.utils import instantiate
from jax import random
from omegaconf import DictConfig, OmegaConf

from dl4bi.core.attention import Attention
from dl4bi.core.mlp import MLP
from dl4bi.core.model_output import MultinomialOutput
from dl4bi.core.train import (
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.tabular import TabularData
from dl4bi.meta_learning.steps import likelihood_train_step, likelihood_valid_step
from dl4bi.meta_learning.utils import cfg_to_run_name


class Mixer(nn.Module):
    proj_out: Callable = MLP([9])
    output_fn: Callable = MultinomialOutput.from_activations
    train_step: Callable = likelihood_train_step
    valid_step: Callable = likelihood_valid_step

    @nn.compact
    def __call__(self, x_ctx: jax.Array, f_ctx: jax.Array, x_test: jax.Array, **kwargs):
        qs, ks, vs = map(lambda x: x[:, None], [x_test, x_ctx, f_ctx])  # add attn head
        f_test, _ = Attention()(qs, ks, vs)
        output = self.proj_out(f_test)[:, 0, :, :]  # remove attn head
        return self.output_fn(output)


@hydra.main("configs/skin_lesions", config_name="default", version_base=None)
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
    optimizer = instantiate(cfg.optimizer)
    model = Mixer()
    # model = instantiate(cfg.model)
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
    batch_size: int = 32,
    num_ctx_min: int = 128,
    num_ctx_max: int = 256,
    num_test: int = 32,
    p_dropout: float = 0.0,
):
    def build_dataloader(name: str):
        data_x = np.load(f"cache/skin_lesions/{name}_x.npy", mmap_mode="r")
        data_f = np.load(f"cache/skin_lesions/{name}_y_mid.npy", mmap_mode="r")
        data_f = jax.nn.one_hot(data_f, 9)
        B, Nc, Nt = batch_size, num_ctx_max, num_test
        (N, D), T = data_x.shape, B * (Nc + Nt)

        def dataloader(rng: jax.Array):
            while True:
                rng_i, rng_d, rng_b, rng = random.split(rng, 4)
                idx = random.choice(rng_i, N, (T,))
                drop = random.bernoulli(rng_d, 1 - p_dropout, (B, Nc + Nt, D))
                x, f = data_x[idx], data_f[idx]
                x = x.reshape(B, Nc + Nt, -1) * drop
                f = f.reshape(B, Nc + Nt, -1)
                d = TabularData(x, f)
                yield d.batch(
                    rng_b,
                    num_ctx_min,
                    num_ctx_max,
                    num_test,
                    test_includes_ctx=False,
                )

        return dataloader

    return map(build_dataloader, ["train", "valid", "test"])


if __name__ == "__main__":
    main()
