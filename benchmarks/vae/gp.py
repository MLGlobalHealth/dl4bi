#!/usr/bin/env python3
import pickle
import shutil
from pathlib import Path
from typing import Optional, Union

import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from hydra.core.hydra_config import HydraConfig
from jax import jit, random
from jax.scipy.stats import norm
from omegaconf import DictConfig, OmegaConf
from sps.gp import GP
from sps.kernels import matern_3_2, periodic, rbf
from sps.priors import Prior
from sps.utils import build_grid
from tqdm import tqdm

from dl4bi.mlp import MLP, gMLP
from dl4bi.vae import DeepChol, PriorCVAE, train_utils


@hydra.main("configs/gp", config_name="default", version_base=None)
def main(cfg: DictConfig):
    d = HydraConfig.get().runtime.choices
    kernel_name, model_name, seed = d["kernel"], d["model"], cfg.seed
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        project=cfg.project,
        name=f"{kernel_name} - {model_name} - seed {seed}",
    )
    rng = random.key(cfg.seed)
    rng_train, rng_eval = random.split(rng)
    s = build_grid([{"start": 0.0, "stop": 1.0, "num": 128}])
    gp, model = instantiate(cfg.kernel), instantiate(cfg.model)
    state = train(rng_train, gp, s, model, cfg.train_num_steps, cfg.valid_interval)
    path = Path(f"results/1D_GP/{kernel_name}/{model_name}-seed-{seed}")
    path.parent.mkdir(parents=True, exist_ok=True)
    validate(
        rng_eval,
        gp,
        s,
        state,
        is_decoder_only=isinstance(model, (DeepChol,)),
        wandb_key="Final Model Samples",
        results_path=path.with_suffix(".pkl"),
    )
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def instantiate(d: Union[dict, DictConfig]):
    """Convenience function to instantiate an object from a config."""
    if isinstance(d, DictConfig):
        d = OmegaConf.to_container(d, resolve=True)
    if "cls" in d:
        cls, kwargs = d["cls"], d.get("kwargs", {})
        for k in kwargs:
            if k == "act_fn":
                kwargs[k] = getattr(nn, kwargs[k])
            elif isinstance(kwargs[k], dict):
                kwargs[k] = instantiate(kwargs[k])
        return globals()[cls](**kwargs)
    elif "func" in d:
        return eval(d["func"])
    return d


def train(
    rng: jax.Array,
    gp: GP,
    s: jax.Array,
    model: nn.Module,
    num_steps: int = 100000,
    validate_every_n: int = 25000,
    batch_size: int = 1024,
    log_every_n: int = 100,
    lr_peak: float = 1e-3,
    lr_pct_warmup: float = 0.3,
    lr_num_cycles: int = 1,
):
    rng_data, rng_params, rng_extra, rng_train = random.split(rng, 4)
    loader = dataloader(rng_data, gp, s, batch_size)
    f, var, ls, period, z = next(loader)
    rngs = {"params": rng_params, "extra": rng_extra}
    x = f if isinstance(model, PriorCVAE) else z  # decoder-only, e.g. DeepChol
    kwargs = model.init(rngs, x, var, ls)
    params = kwargs.pop("params")
    param_count = nn.tabulate(model, rngs)(x, var, ls)
    learning_rate_fn = create_learning_rate_fn(
        num_steps,
        lr_peak,
        lr_pct_warmup,
        lr_num_cycles,
    )
    state = train_utils.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.yogi(learning_rate_fn),
        kwargs=kwargs,
    )
    print(f"{model}\n\n{param_count}")
    is_decoder_only = False
    train_step = train_utils.elbo_train_step
    if isinstance(model, (DeepChol,)):
        is_decoder_only = True
        train_step = train_utils.mse_train_step
    losses = np.zeros((num_steps,))
    for i in (pbar := tqdm(range(num_steps), unit="batch", dynamic_ncols=True)):
        rng_step, rng_train = random.split(rng_train)
        batch = next(loader)
        state, losses[i] = train_step(rng_step, state, batch)
        if i > 0 and i % log_every_n == 0:
            avg = jnp.mean(losses[i - log_every_n : i])
            pbar.set_postfix(loss=f"{avg:.3f}")
            wandb.log({"loss": avg})
        if i > 0 and i % validate_every_n == 0:
            rng_valid, rng_train = random.split(rng_train)
            validate(rng_valid, gp, s, state, is_decoder_only, wandb_key=f"Step {i}")
    return state


def dataloader(rng, gp, s, batch_size=1024, approx=True):
    while True:
        rng_batch, rng = random.split(rng)
        yield gp.simulate(rng, s, batch_size, approx)


def create_learning_rate_fn(
    num_steps: int,
    peak_lr: float,
    pct_warmup: float = 0.3,
    num_cycles: int = 1,
):
    """Create an n-cycle cosine annealing schedule."""
    n = num_steps // num_cycles
    sched = optax.cosine_onecycle_schedule(n, peak_lr, pct_start=pct_warmup)
    boundaries = n * jnp.arange(1, num_cycles)
    return optax.join_schedules([sched] * num_cycles, boundaries)


def custom_learning_rate_fn(num_steps: int, peak_lr: float):
    """Create a 3-cycle cosine annealing schedule.

    There are two cosine schedules each consisting of a quarter of `num_steps`
    and then a third single cosine schedule consisting of half of `num_steps`.
    """
    q, r = num_steps // 4, num_steps % 4
    q_sched = optax.cosine_onecycle_schedule(q, peak_lr, pct_start=0.2)
    h_sched = optax.cosine_onecycle_schedule(2 * q + r, peak_lr, pct_start=0.2)
    boundaries = [0, q, 2 * q]
    return optax.join_schedules([q_sched, q_sched, h_sched], boundaries)


def validate(
    rng: jax.Array,
    gp: GP,
    s: jax.Array,
    state: train_utils.TrainState,
    is_decoder_only: bool = False,
    num_batches: int = 5000,
    num_plots: int = 16,
    wandb_key: str = "",
    results_path: Optional[Path] = None,
):
    rng_data, rng_extra, rng_plots = random.split(rng, 3)
    loader = dataloader(rng_data, gp, s)
    losses = np.zeros((num_batches,))
    results = []
    for i in (pbar := tqdm(range(num_batches), unit="batch", dynamic_ncols=True)):
        batch = next(loader)
        f, var, ls, period, z = batch
        params = {"params": state.params, **state.kwargs}
        rngs = {"extra": rng_extra}
        if is_decoder_only:
            f_hat = jit(state.apply_fn)(params, z, var, ls)
            losses[i] = optax.squared_error(f_hat, f.squeeze()).mean()
        else:
            f_hat, z_mu, z_std = jit(state.apply_fn)(params, f, var, ls, rngs=rngs)
            kl_div = -jnp.log(z_std) + (z_std**2 + z_mu**2 - 1) / 2
            logp = norm.logpdf(f, f_hat, 1.0).mean()
            losses[i] = -logp + kl_div.mean()
        if results_path:
            b = [np.array(v) for v in batch]
            p = np.array(f_hat)
            results += [(b, p)]
    loss = losses.mean()
    print(f"validation loss: {loss:.3f}")
    wandb.log({"validation_loss": loss})
    log_plots(rng_plots, wandb_key, num_plots, batch, s, f_hat)
    if results_path:
        with open(results_path, "wb") as f:
            pickle.dump(results, f)


def log_plots(
    rng: jax.Array,
    wandb_key: str,
    num_plots: int,
    batch: tuple,
    s: jax.Array,
    f_hat: jax.Array,
):
    """Logs `num_plots` from the given batch."""
    (f, var, ls, period, z), s_flat = batch, s.squeeze()
    sample_paths = []
    for i in random.choice(rng, f.shape[0], (num_plots,), replace=False):
        plt.plot(s_flat, f[i].squeeze().T, color="black")
        plt.plot(s_flat, f_hat[i].squeeze().T, color="red")
        title = f"Sample {i} (var: {var[0]:0.2f}, ls: {ls[0]:0.2f})"
        path = f"/tmp/{title}.png"
        sample_paths += [path]
        plt.title(title)
        plt.savefig(path, dpi=150)
        plt.clf()
    wandb.log({wandb_key: [wandb.Image(p) for p in sample_paths]})


def save_ckpt(state: train_utils.TrainState, cfg: DictConfig, path: Path):
    """Saves a checkpoint."""
    shutil.rmtree(path, ignore_errors=True)
    ckptr = ocp.Checkpointer(ocp.CompositeCheckpointHandler("state", "config"))
    cfg_d = OmegaConf.to_container(cfg, resolve=True)
    ckptr.save(
        path.absolute(),
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(state),
            config=ocp.args.JsonSave(cfg_d),
        ),
    )


def load_ckpt(path: Path):
    """Loads a checkpoint."""
    ckptr = ocp.Checkpointer(ocp.CompositeCheckpointHandler("state", "config"))
    # restore config and use it to create model template
    ckpt = ckptr.restore(path, args=ocp.args.Composite(config=ocp.args.JsonRestore()))
    cfg = OmegaConf.create(ckpt["config"])
    num_steps, batch_size = 100000, 1024
    lr_peak, lr_pct_warmup, lr_num_cycles = 1e-3, 0.1, 1
    rng = random.key(42)
    rng_gp, rng_params, rng_extra = random.split(rng, 3)
    s = build_grid([{"start": 0, "stop": 1.0, "num": 128}])
    gp, model = instantiate(cfg.kernel), instantiate(cfg.model)
    f, var, ls, period, z = gp.simulate(rng_gp, s, batch_size)
    x = f if isinstance(model, PriorCVAE) else z  # decoder-only, e.g. DeepChol
    rngs = {"params": rng_params, "extra": rng_extra}
    kwargs = model.init(rngs, x, var, ls)
    params = kwargs.pop("params")
    learning_rate_fn = create_learning_rate_fn(
        num_steps,
        lr_peak,
        lr_pct_warmup,
        lr_num_cycles,
    )
    state = train_utils.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.yogi(learning_rate_fn),
        kwargs=kwargs,
    )
    ckpt = ckptr.restore(
        path, args=ocp.args.Composite(state=ocp.args.StandardRestore(state))
    )
    return ckpt["state"], model, cfg


if __name__ == "__main__":
    main()
