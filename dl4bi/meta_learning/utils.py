import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from jax import random
from omegaconf import DictConfig
from tqdm import tqdm

from ..core.train import TrainState, load_ckpt
from .data.spatial import SpatialBatch


def first_shape(arrays: Sequence[Union[jax.Array, None]]) -> tuple:
    for array in arrays:
        if array is not None:
            return array.shape
    return ()


def cfg_to_run_name(cfg: DictConfig):
    return cfg.model._target_.split(".")[-1]


def load_ckpts(
    dir: Union[str, Path],
    only_regex: Union[str, re.Pattern] = r".*",
    exclude_regex: Union[str, re.Pattern] = "$^",
):
    """Loads all checkpoints in a given base dir."""
    ckpt = {}
    if isinstance(only_regex, str):
        only_regex = re.compile(only_regex, re.IGNORECASE)
    if isinstance(exclude_regex, str):
        exclude_regex = re.compile(exclude_regex, re.IGNORECASE)
    for p in Path(dir).glob("*.ckpt"):
        if only_regex.match(str(p)) and not exclude_regex.match(str(p)):
            state, tmp_cfg = load_ckpt(p)
            ckpt[cfg_to_run_name(tmp_cfg)] = {"state": state, "cfg": tmp_cfg}
    return ckpt


def regression_to_rgb(f: jax.Array):
    return jnp.clip(f / 2 + 0.5, 0, 1)  # [-1, 1] => [0, 1]


def wandb_2d_img_callback(
    step: int,
    rng_step: int,
    state: TrainState,
    batch: SpatialBatch,
    extra: dict,
    filename_prefix: Optional[str] = "step",
    transform_model_output: Callable = lambda x: (x.mu, x.std),
    **kwargs,
):
    """Logs `num_plots` from the given batch for 2D GPs."""
    rng_dropout, rng_extra = random.split(rng_step)
    output = state.apply_fn(
        {"params": state.params, **state.kwargs},
        **batch,
        rngs={"dropout": rng_dropout, "extra": rng_extra},
    )
    if isinstance(output, tuple):
        output, _ = output  # throw away latent samples
    f_pred, f_std = transform_model_output(output)
    path = f"/tmp/{filename_prefix}_{step}_{datetime.now().isoformat()}.png"
    fig = batch.plot_2d(f_pred, f_std, **kwargs)
    fig.savefig(path)
    plt.close(fig)
    wandb.log({f"Step {step}": wandb.Image(path)})


def x_to_none(x: jax.Array):
    return None


def save_batches_for_tabpfn(
    rng: jax.Array,
    dataloader: Callable,
    num_steps: int,
    path: Path,
):
    rng_data, rng = random.split(rng)
    print("Saving batches for TabPFN")
    pbar = tqdm(
        dataloader(rng_data),
        total=num_steps,
        unit=" batches",
        leave=False,
        dynamic_ncols=True,
    )
    samples = []
    for i, batch in enumerate(pbar):
        if i >= num_steps:  # for infinite dataloaders
            break
        samples.append(batch.to_xy())
    np.save(path, samples, allow_pickle=True)
