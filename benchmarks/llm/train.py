#!/usr/bin/env python3
import os
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import wandb
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from hydra.core.hydra_config import HydraConfig
from jax import jit, random
from jax.experimental import enable_x64
from omegaconf import DictConfig, OmegaConf
from optax.losses import softmax_cross_entropy_with_integer_labels
from orbax.checkpoint import PyTreeCheckpointer
from tqdm import tqdm

from dl4bi.llm import GPT

# https://jax.readthedocs.io/en/latest/gpu_performance_tips.html#code-generation-flags
# https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/GPU_performance.md
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_async_collectives=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
)

# TODO(danj):
# Parallelize across devices & test out HPC
# Add rng to train state
# Generate function


@hydra.main("configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    choices = HydraConfig.get().runtime.choices
    default_name = choices["model"] + "_" + choices["data"]
    run_name = cfg.get("name", cfg.get("name", default_name))
    path = Path(f"results/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=cfg.get("name", run_name),
        project=cfg.project,
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    train_dataloader, valid_dataloader = build_dataloaders(
        cfg.data.path,
        cfg.data.batch_size,
        cfg.model.num_context_window,
    )
    lr_schedule = optax.warmup_cosine_decay_schedule(
        cfg.data.lr_min,
        cfg.data.lr_peak,
        cfg.data.lr_warmup_num_steps,
        cfg.data.train_num_steps,
        cfg.data.lr_min,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.data.clip_max_norm),
        optax.adamw(
            lr_schedule,
            cfg.data.beta_1,
            cfg.data.beta_2,
            weight_decay=cfg.data.weight_decay,
            # only weight decay 2D params, i.e. ignore scale & bias
            mask=lambda params: jtu.tree_map(lambda x: x.ndim != 1, params),
        ),
    )
    optimizer = optax.MultiSteps(optimizer, cfg.data.grad_accum_steps)
    model = GPT(**cfg.model)
    callbacks = []
    if "ckpt_interval" in cfg.data:
        ckpt_cb = Callback(
            lambda step, rng_step, state, *_: save_ckpt(
                state, cfg, path.with_suffix(".ckpt")
            ),
            cfg.data.ckpt_interval,
        )
        callbacks += [ckpt_cb]
    state = train(
        rng_train,
        model,
        optimizer,
        train_dataloader,
        valid_dataloader,
        cfg.data.train_num_steps,
        cfg.data.valid_num_steps,
        cfg.data.valid_interval,
        cfg.data.grad_accum_steps,
        callbacks=callbacks,
    )
    loss = evaluate(rng_test, state, valid_dataloader, cfg.data.valid_num_steps)
    wandb.log({"test_loss": loss})
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


@dataclass
class Callback:
    fn: Callable  # (step, rng_step, state, batch) -> None
    interval: int  # apply every interval of train_num_steps


def train(
    rng: jax.Array,
    model: nn.Module,
    optimizer: optax.MultiSteps,
    train_dataloader: Callable,
    valid_dataloader: Callable,
    train_num_steps: int = 100000,
    valid_num_steps: Optional[int] = None,
    valid_interval: int = 25000,
    grad_accum_steps: int = 1,
    log_loss_interval: int = 100,
    callbacks: list[Callback] = [],
):
    rng_data, rng_init, rng_step = random.split(rng, 3)
    batches = train_dataloader(rng_data)
    token_ids, _ = next(batches)
    params = model.init(rng_init, token_ids)
    param_count = nn.tabulate(model, rng_init)(token_ids)
    print(param_count)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    losses = []
    train_loss, valid_loss = float("inf"), float("inf")
    pbar = tqdm(range(1, train_num_steps + 1), unit="step", dynamic_ncols=True)
    for step in pbar:
        batch_losses = []
        for k in range(grad_accum_steps):
            batch = next(batches)
            state, loss = train_step(state, batch)
            batch_losses += [loss]
        losses += [np.mean(batch_losses)]
        if step % log_loss_interval == 0:
            train_loss = np.mean(losses[step - log_loss_interval : step])
            wandb.log({"train_loss": train_loss})
        if step % valid_interval == 0:
            rng_valid, rng_step = random.split(rng_step)
            valid_loss = evaluate(rng_valid, state, valid_dataloader, valid_num_steps)
            wandb.log({"valid_loss": valid_loss})
        for cbk in callbacks:
            if step % cbk.interval == 0:
                cbk.fn(step, rng_step, state, batch)  # only last grad accum batch
        pbar.set_postfix(
            {"train_loss": f"{train_loss:.3f}", "valid_loss": f"{valid_loss:.3f}"}
        )
    return state


def build_dataloaders(
    path: str,
    batch_size: int = 16,
    num_context_window: int = 1024,
):
    B, L = batch_size, num_context_window

    def build_dataloader(path: Path):
        dataset = np.memmap(path, mode="r", dtype=np.uint16)
        N = len(dataset) - L  # only calculate once

        def dataloader(rng: jax.Array):
            while True:
                rng_idx, rng = random.split(rng)
                with enable_x64():  # uint32 only goes up to 4.3B
                    idx = random.randint(rng_idx, (B,), 0, N)
                    token_ids = np.stack([dataset[i : i + L] for i in idx])
                    next_token_ids = np.stack([dataset[j : j + L] for j in idx + 1])
                yield token_ids, next_token_ids

        return dataloader

    return (
        build_dataloader(Path(path) / "train.npy"),
        build_dataloader(Path(path) / "valid.npy"),
    )


@partial(jit, donate_argnames=["state"])
def train_step(state: TrainState, batch: tuple):
    def loss_fn(params):
        token_ids, next_token_ids = batch
        logits = state.apply_fn(params, token_ids).astype(jnp.float32)
        return softmax_cross_entropy_with_integer_labels(logits, next_token_ids).mean()

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


def evaluate(
    rng: jax.Array,
    state: TrainState,
    dataloader: Callable,
    num_steps: Optional[float] = None,
):
    num_steps = num_steps or float("inf")
    pbar = tqdm(
        dataloader(rng),
        total=num_steps,
        unit=" batches",
        leave=False,
        dynamic_ncols=True,
    )
    losses = []
    for i, batch in enumerate(pbar):
        # early stopping for infinite dataloaders
        if i >= num_steps:
            break
        token_ids, next_token_ids = batch
        logits = jit(state.apply_fn)(state.params, token_ids).astype(jnp.float32)
        losses += [
            softmax_cross_entropy_with_integer_labels(logits, next_token_ids).mean()
        ]
    return np.mean(losses)


def save_ckpt(state: TrainState, cfg: DictConfig, path: Path):
    "Save a checkpoint."
    shutil.rmtree(path, ignore_errors=True)
    ckptr = PyTreeCheckpointer()
    ckpt = {"state": state, "config": OmegaConf.to_container(cfg, resolve=True)}
    save_args = orbax_utils.save_args_from_target(ckpt)
    ckptr.save(path.absolute(), ckpt, save_args=save_args)


def load_ckpt(path: Path):
    "Load a checkpoint."
    ckptr = PyTreeCheckpointer()
    ckpt = ckptr.restore(path.absolute())
    cfg = OmegaConf.create(ckpt["config"])
    model = GPT(**cfg.model)
    return cfg, model, ckpt["state"]["params"]


if __name__ == "__main__":
    main()
