import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import flax
import flax.linen as nn
import jax
import numpy as np
import optax
import wandb
from flax.core import FrozenDict
from flax.training import orbax_utils, train_state
from hydra.utils import instantiate
from jax import random
from omegaconf import DictConfig, OmegaConf
from orbax.checkpoint import PyTreeCheckpointer
from tqdm import tqdm


@flax.struct.dataclass
class TrainState(train_state.TrainState):
    # kwargs stores any extra information associated with training,
    # i.e. batch norm stats or fixed (random) projections
    kwargs: FrozenDict = FrozenDict({})


@dataclass
class Callback:
    fn: Callable  # (step, rng_step, state, batch, extra) -> None
    interval: int  # apply every interval of train_num_steps


def train(
    rng: jax.Array,
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    train_step: Callable,
    train_num_steps: int,
    train_dataloader: Callable,
    valid_step: Optional[Callable] = None,
    valid_interval: Optional[int] = None,
    valid_num_steps: Optional[int] = None,
    valid_dataloader: Optional[Callable] = None,
    valid_monitor_metric: str = "NLL",
    early_stop_patience: Optional[int] = None,
    callbacks: list[Callback] = [],
    callback_dataloader: Optional[Callable] = None,
    log_loss_interval: int = 100,
    return_state: str = "last",  # best, last, both
    state: Optional[TrainState] = None,
):
    rng_data, rng_params, rng_extra, rng_train = random.split(rng, 4)
    batches = train_dataloader(rng_data)
    batch = next(batches)
    rngs = {"params": rng_params, "extra": rng_extra}
    kwargs = model.init(rngs, **batch)
    params = kwargs.pop("params")
    param_count = nn.tabulate(model, rngs)(**batch)
    print(param_count)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params if state is None else state.params,
        kwargs=kwargs if state is None else state.kwargs,
        tx=optimizer,
    )
    losses = []
    patience = 0
    best_state = state
    early_stop_patience = early_stop_patience or train_num_steps
    train_loss, metric, best_metric = float("inf"), float("inf"), float("inf")
    pbar = tqdm(range(1, train_num_steps + 1), unit="batch", dynamic_ncols=True)
    postfix = {"Train Loss": f"{train_loss:0.4f}"}
    for i in pbar:
        batch = next(batches)
        rng_train_step, rng_train = random.split(rng_train)
        state, loss = train_step(rng_train_step, state, batch)
        losses += [loss]
        if i % log_loss_interval == 0:
            train_loss = np.mean(losses)
            losses = []
            wandb.log({"Train Loss": train_loss})
        postfix["Train Loss"] = f"{train_loss:.4f}"
        if valid_interval and i % valid_interval == 0:
            rng_valid, rng_train = random.split(rng_train)
            metrics = evaluate(
                rng_valid,
                state,
                valid_step,
                valid_dataloader,
                valid_num_steps,
            )
            metric = metrics[valid_monitor_metric]
            postfix[f"Valid {valid_monitor_metric}"] = f"{metric:0.4f}"
            wandb.log({f"Valid {m}": v for m, v in metrics.items()})
            patience += 1
            if metric < best_metric:
                patience = 0
                best_metric = metric
                best_state = state
            if patience >= early_stop_patience:
                both = (best_state, state)
                return {"best": best_state, "last": state, "both": both}[return_state]
        for cbk in callbacks:
            if i % cbk.interval == 0:
                extra = None
                if callback_dataloader is not None:
                    batch = next(callback_dataloader(rng_train_step))
                    batch, extra = batch if isinstance(batch, tuple) else (batch, None)
                cbk.fn(i, rng_train_step, state, batch, extra)
        pbar.set_postfix(postfix)
    both = (best_state, state)
    return {"best": best_state, "last": state, "both": both}[return_state]


def evaluate(
    rng: jax.Array,
    state: TrainState,
    valid_step: Callable,
    dataloader: Callable,
    num_steps: Optional[int],
):
    rng_data, rng = random.split(rng)
    num_steps = num_steps or float("inf")
    pbar = tqdm(
        dataloader(rng_data),
        total=num_steps,
        unit=" batches",
        leave=False,
        dynamic_ncols=True,
    )
    metrics = defaultdict(list)
    for i, batch in enumerate(pbar):
        rng_step, rng = random.split(rng)
        if i >= num_steps:  # for infinite dataloaders
            break
        m = valid_step(rng_step, state, batch)
        for k, v in m.items():
            metrics[k] += [v]
    return {k: np.mean(v) for k, v in metrics.items()}


def save_ckpt(state: TrainState, cfg: DictConfig, path: Path):
    "Save a checkpoint."
    shutil.rmtree(path, ignore_errors=True)
    ckptr = PyTreeCheckpointer()
    ckpt = {"state": state, "config": OmegaConf.to_container(cfg, resolve=True)}
    save_args = orbax_utils.save_args_from_target(ckpt)
    ckptr.save(path.absolute(), ckpt, save_args=save_args)


def load_ckpt(path: Union[str, Path]):
    "Load a checkpoint."
    if not isinstance(path, Path):
        path = Path(path)
    ckptr = PyTreeCheckpointer()
    ckpt = ckptr.restore(path.absolute())
    cfg = OmegaConf.create(ckpt["config"])
    model = instantiate(cfg.model)
    state = TrainState.create(
        apply_fn=model.apply,
        # TODO(danj): reload optimizer state
        tx=optax.yogi(cosine_annealing_lr()),
        params=ckpt["state"]["params"],
        kwargs=ckpt["state"]["kwargs"],
    )
    return state, cfg


def cosine_annealing_lr(
    num_steps: int = 100000,
    lr_max: float = 1e-3,
    lr_min: float = 1e-4,
):
    return optax.cosine_decay_schedule(lr_max, num_steps, lr_min)
