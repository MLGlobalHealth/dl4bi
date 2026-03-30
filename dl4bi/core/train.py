"""Training loops, evaluation helpers, and checkpoint utilities."""

import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.core import FrozenDict
from flax.training import orbax_utils, train_state
from hydra.utils import instantiate
from jax import jit, random
from omegaconf import DictConfig, OmegaConf
from orbax.checkpoint import PyTreeCheckpointer
from tqdm import tqdm


@flax.struct.dataclass
class TrainState(train_state.TrainState):
    """Training state extended with non-parameter model variables."""

    # kwargs stores any extra information associated with training,
    # i.e. batch norm stats or fixed (random) projections
    kwargs: FrozenDict = FrozenDict({})


@dataclass
class Callback:
    """Periodic callback invoked during training."""

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
    callbacks: Optional[list[Callback]] = None,
    callback_dataloader: Optional[Callable] = None,
    log_loss_interval: int = 100,
    return_state: str = "last",  # best, last, both
    state: Optional[TrainState] = None,
):
    """Train a Flax model with optional validation and callbacks.

    Args:
        rng: Root PRNG key.
        model: Model to initialize and train.
        optimizer: Optax optimizer.
        train_step: Step function returning ``(state, loss)``.
        train_num_steps: Number of training steps to run.
        train_dataloader: Callable that yields training batches.
        valid_step: Optional validation step function.
        valid_interval: Validation cadence in training steps.
        valid_num_steps: Number of validation batches to consume.
        valid_dataloader: Callable that yields validation batches.
        valid_monitor_metric: Metric name used for early stopping.
        early_stop_patience: Number of validation rounds without improvement.
        callbacks: Periodic callbacks to execute during training.
        callback_dataloader: Optional dataloader for callback batches.
        log_loss_interval: Training-loss logging cadence.
        return_state: Which checkpointed state to return: ``"best"``, ``"last"``,
            or ``"both"``.
        state: Optional existing state to resume from.

    Returns:
        The requested trained state, or both best and last states.
    """
    callbacks = callbacks or []
    rng_data, rng_params, rng_extra, rng_train, rng_valid = random.split(rng, 5)
    batches = train_dataloader(rng_data)
    batch = next(batches)
    rngs = {"params": rng_params, "extra": rng_extra}
    kwargs = model.init(rngs, **batch)
    params = kwargs.pop("params")
    state = TrainState.create(
        apply_fn=model.apply,
        params=params if state is None else state.params,
        kwargs=kwargs if state is None else state.kwargs,
        tx=optimizer,
    )
    # TODO(danj): FLOPS returning 0 -- https://github.com/google/flax/issues/4023s
    # Remove manual GFLOPS count below when fixed
    param_count = nn.tabulate(model, rngs, compute_flops=True, compute_vjp_flops=True)(
        **batch
    )
    infer_flops, train_flops = estimate_flops(rng_train, state, train_step, batch)
    bold, reset = "\033[1m", "\033[0m"
    print(param_count)
    print(f"{bold}Estimated Infer GFLOPS: {infer_flops / 1.0e9:g}{reset}")
    print(f"{bold}Estimated Train GFLOPS: {train_flops / 1.0e9:g}\n\n{reset}")
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
            rng_valid_step, rng_valid = random.split(rng_valid)
            metrics = evaluate(
                rng_valid_step,
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


def estimate_flops(rng, state, train_step, batch):
    """Estimate inference and training FLOPs for a single batch."""
    infer_cost = jit(infer).lower(rng, state, batch).compile().cost_analysis()
    train_cost = jit(train_step).lower(rng, state, batch).compile().cost_analysis()
    return infer_cost["flops"], train_cost["flops"]


@jit
def infer(rng, state, batch):
    """Run model inference with the stored parameters and variables."""
    return state.apply_fn(
        {"params": state.params, **state.kwargs},
        **batch,
        training=False,
        rngs={"extra": rng},
    )


def evaluate(
    rng: jax.Array,
    state: TrainState,
    valid_step: Callable,
    dataloader: Callable,
    num_steps: Optional[int],
):
    """Aggregate validation metrics over a dataloader."""
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


def load_ckpt(path: Union[str, Path], override_cfg: Optional[DictConfig] = None):
    "Load a checkpoint."
    if not isinstance(path, Path):
        path = Path(path)
    ckptr = PyTreeCheckpointer()
    ckpt = ckptr.restore(path.absolute())
    cfg = OmegaConf.create(ckpt["config"])
    if override_cfg is not None:
        cfg = override_cfg
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
    """Build a cosine-decay learning-rate schedule."""
    return optax.cosine_decay_schedule(lr_max, num_steps, lr_min)
