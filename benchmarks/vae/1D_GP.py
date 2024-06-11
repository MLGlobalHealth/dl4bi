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
from tqdm import tqdm

from dsp.core import *  # noqa: F403
from dsp.meta_regression import (
    DeepChol,
    PiVAE,
    PriorCVAE,
)


@hydra.main("configs/1D_GP", version_base=None)
def main(cfg: DictConfig):
    d = HydraConfig.get().runtime.choices
    kernel_name, model_name, seed = d["kernel"], d["model"], cfg.seed
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if "wandb" in cfg else "disabled",
        name=f"{kernel_name} - {model_name} - seed {seed}",
    )
    rng = random.key(cfg.seed)
    rng_train, rng_eval = random.split(rng)
    gp, model = instantiate(cfg.kernel), instantiate(cfg.model)
    state = train(rng_train, gp, model)
    path = Path(f"results/1D_GP/{kernel_name}/{model_name}-seed-{seed}")
    path.parent.mkdir(parents=True, exist_ok=True)
    validate(
        rng_eval,
        gp,
        state,
        wandb_key="Final Model Samples",
        results_path=path.with_suffix(".pkl"),
    )
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def train(
    rng: jax.Array,
    gp: GP,
    model: nn.Module,
    num_steps: int = 200000,
    validate_every_n: int = 50000,
    log_every_n: int = 100,
    lr_peak: float = 1e-3,
    lr_pct_warmup: float = 0.3,
    lr_num_cycles: int = 1,
):
    rng_data, rng_params, rng_latent_z, rng_train = random.split(rng, 4)
    loader = dataloader(rng_data, gp)
    s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, var, ls = next(
        loader
    )
    kwargs = model.init(
        {"params": rng_params, "latent_z": rng_latent_z},
        s_ctx,
        f_ctx,
        s_test,
        valid_lens_ctx,
        valid_lens_test,
    )
    params = kwargs.pop("params")
    learning_rate_fn = create_learning_rate_fn(
        num_steps,
        lr_peak,
        lr_pct_warmup,
        lr_num_cycles,
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.yogi(learning_rate_fn),
        kwargs=kwargs,
    )
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"{model}\n\nParam count: {param_count}")
    train_step = train_steps.train_step
    if isinstance(model, (NP, ANP)):
        train_step = train_steps.npf_elbo_train_step
    elif isinstance(model, (TNPND,)):
        train_step = train_steps.train_step_tril_cov
    losses = np.zeros((num_steps,))
    for i in (pbar := tqdm(range(num_steps), unit="batch")):
        rng_step, rng_train = random.split(rng_train)
        batch = next(loader)
        state, losses[i] = train_step(rng_step, state, batch)
        if i > 0 and i % log_every_n == 0:
            avg = jnp.mean(losses[i - log_every_n : i])
            pbar.set_postfix(loss=f"{avg:.3f}")
            wandb.log({"loss": avg})
        if i > 0 and i % validate_every_n == 0:
            rng_valid, rng_train = random.split(rng_train)
            validate(rng_valid, gp, state, wandb_key=f"Step {i}")
    return state


def validate(
    rng: jax.Array,
    gp: GP,
    state: TrainState,
    num_batches: int = 5000,
    num_plots: int = 16,
    wandb_key: str = "",
    results_path: Optional[Path] = None,
):
    rng_data, rng_latent_z, rng_plots = random.split(rng, 3)
    loader = dataloader(rng_data, gp)
    losses = np.zeros((num_batches,))
    results = []
    for i in (pbar := tqdm(range(num_batches), unit="batch")):
        batch = next(loader)
        s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, var, ls = batch
        mask_test = mask_from_valid_lens(s_test.shape[1], valid_lens_test)
        f_mu, f_std, *_ = jit(state.apply_fn)(
            {"params": state.params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            valid_lens_test,
            rngs={"latent_z": rng_latent_z},  # used by NP family
        )
        if f_mu.shape == f_std.shape:  # f_std is independent/diagonal
            losses[i] = -norm.logpdf(f_test, f_mu, f_std).mean(where=mask_test)
        else:  # f_std is a lower triangular covariance matrix
            # WARNING: This ignores `valid_lens_test` because
            # mvn_logpdf_tril_cov does yet support masks with `where`.
            B, L_test, _ = f_test.shape
            f_test_flat, f_mu_flat = f_test.reshape(B, -1), f_mu.reshape(B, -1)
            nll = -mvn_logpdf_tril_cov(f_test_flat, f_mu_flat, f_std).mean()
            losses[i] = nll / L_test
        if results_path:
            b = [np.array(v) for v in batch]
            p = [np.array(v) for v in [f_mu, f_std]]
            results += [(b, p)]
    loss = losses.mean()
    print(f"validation loss: {loss:.3f}")
    wandb.log({"validation_loss": loss})
    log_plots(rng_plots, wandb_key, num_plots, batch, f_mu, f_std)
    if results_path:
        with open(results_path, "wb") as f:
            pickle.dump(results, f)


def log_plots(
    rng: jax.Array,
    wandb_key: str,
    num_plots: int,
    batch: tuple,
    f_mu: jax.Array,
    f_std: jax.Array,
):
    """Logs `num_plots` from the given batch."""
    s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, var, ls = batch
    batch_size = s_test.shape[0]
    sample_paths = []
    for i in random.choice(rng, batch_size, (num_plots,), replace=False):
        f_std_i = f_std[i].squeeze()
        # TODO(danj): is this legitimate?
        if f_mu.shape != f_std.shape:  # f_std L in Sigma=LL^T
            f_std_i = jnp.diag(f_std_i @ f_std_i.T)
        sample_path = plot_posterior_predictive(
            i,
            s_ctx[i].squeeze(),
            f_ctx[i].squeeze(),
            valid_lens_ctx[i],
            s_test[i].squeeze(),
            f_test[i].squeeze(),
            valid_lens_test[i],
            f_mu[i].squeeze(),
            f_std_i,
            var,
            ls,
        )
        sample_paths += [sample_path]
    wandb.log({wandb_key: [wandb.Image(p) for p in sample_paths]})


def plot_posterior_predictive(
    id: int,
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    valid_len_ctx: jax.Array,
    s_test: jax.Array,
    f_test: jax.Array,
    valid_len_test: jax.Array,
    f_mu: jax.Array,
    f_std: jax.Array,
    var: np.ndarray,
    ls: np.ndarray,
    hdi_prob=0.95,
):
    """Plots the posterior predictive alongside true values."""
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    f_lower, f_upper = f_mu - z_score * f_std, f_mu + z_score * f_std
    s_ctx, f_ctx = s_ctx[:valid_len_ctx], f_ctx[:valid_len_ctx]
    s_test, f_test = s_test[:valid_len_test], f_test[:valid_len_test]
    f_mu = f_mu[:valid_len_test]
    f_lower, f_upper = f_lower[:valid_len_test], f_upper[:valid_len_test]
    idx = jnp.argsort(s_test)
    plt.plot(s_test[idx], f_test[idx], color="black")
    plt.plot(s_test[idx], f_mu[idx], color="steelblue")
    plt.scatter(s_ctx, f_ctx, color="black")
    plt.fill_between(
        s_test[idx],
        f_lower[idx],
        f_upper[idx],
        alpha=0.4,
        color="steelblue",
        interpolate=True,
    )
    ax = plt.gca()
    ax.set_xlabel("s")
    ax.set_ylabel("f")
    title = f"Sample {id} (var: {var[0]:0.2f}, ls: {ls[0]:0.2f})"
    path = f"/tmp/{title}.png"
    plt.title(title)
    plt.savefig(path, dpi=150)
    plt.clf()
    return path


def _results_path(kernel: str, model: str, seed: int):
    path = Path(f"results/1D_GP/{kernel}/{model}-seed-{seed}.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.absolute()


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


def dataloader(rng: jax.Array, gp: GP):
    """Generates batches of GP samples."""
    batch_size = 16
    s_min, s_max = -2, 2
    obs_noise = 0.1
    num_ctx_min, num_ctx_max, num_linear = 3, 50, 100
    s_linear = jnp.linspace(s_min, s_max, num_linear)
    valid_lens_test = jnp.repeat(num_ctx_max + num_linear, batch_size)

    @jit
    def gen_batch(rng: jax.Array):
        rng_s_random, rng_valid_lens_ctx, rng_gp, rng_eps, rng = random.split(rng, 5)
        s_random = random.uniform(rng_s_random, (num_ctx_max,), float, s_min, s_max)
        s = jnp.hstack([s_random, s_linear])[:, None]  # [num_ctx_max + num_linear, 1]
        var, ls, _z, f = gp.simulate(rng_gp, s, batch_size)
        valid_lens_ctx = random.randint(
            rng_valid_lens_ctx, (batch_size,), num_ctx_min, num_ctx_max
        )
        s = jnp.repeat(s[None, ...], batch_size, axis=0)
        f_noisy = f + obs_noise * random.normal(rng_eps, f.shape)
        return s, f_noisy, valid_lens_ctx, s, f, valid_lens_test, var, ls

    while True:
        rng_batch, rng = random.split(rng)
        yield gen_batch(rng_batch)


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


def save_ckpt(state: TrainState, cfg: DictConfig, path: Path):
    """Saves a checkpoint.

    .. warning::
        This uses the current runtime settings in Hydra to resolve the
        checkpoint path.
    """
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
    """Loads a checkpoint.

    .. warning::
        This uses the current runtime settings in Hydra to resolve the
        checkpoint path.
    """
    ckptr = ocp.Checkpointer(ocp.CompositeCheckpointHandler("state", "config"))
    # restore config and use it to create model template
    ckpt = ckptr.restore(path, args=ocp.args.Composite(config=ocp.args.JsonRestore()))
    cfg = OmegaConf.create(ckpt["config"])
    model = instantiate(cfg.model)
    num_steps = 100000
    lr_peak, lr_pct_warmup, lr_num_cycles = 1e-3, 0.1, 1
    B, L, D = 4, 10, 1  # these are arbitrary
    s = f = jnp.zeros((B, L, D))
    valid_lens = jnp.repeat(L, B)
    key = random.key(42)
    kwargs = model.init(key, s, f, s, valid_lens, valid_lens)
    params = kwargs.pop("params")
    learning_rate_fn = create_learning_rate_fn(
        num_steps,
        lr_peak,
        lr_pct_warmup,
        lr_num_cycles,
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.yogi(learning_rate_fn),
        kwargs=kwargs,
    )
    ckpt = ckptr.restore(
        path, args=ocp.args.Composite(state=ocp.args.StandardRestore(state))
    )
    return ckpt["state"], cfg


if __name__ == "__main__":
    main()
