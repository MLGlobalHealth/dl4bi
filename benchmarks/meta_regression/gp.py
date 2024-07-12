#!/usr/bin/env python3
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from hydra.core.hydra_config import HydraConfig
from jax import jit, random
from jax.scipy.stats import norm
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid

from dsp.meta_regression.train_utils import (
    Callback,
    TrainState,
    cosine_annealing_lr,
    instantiate,
    save_ckpt,
    train,
    validate,
)


@hydra.main("configs/gp", version_base=None)
def main(cfg: DictConfig):
    d = HydraConfig.get().runtime.choices
    exp, kernel, model_name = d["exp"], d["kernel"], d["model"]
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if "wandb" in cfg else "disabled",
        name=cfg.get("name", f"{exp} {kernel} - {model_name} - seed {cfg.seed}"),
        project="SPTx - GPs",
    )
    rng = random.key(cfg.seed)
    rng_train, rng_valid = random.split(rng)
    dataloader = build_dataloader(cfg.exp, cfg.kernel)
    train_num_steps, valid_num_steps = 100000, 5000
    valid_interval, plot_interval = 25000, 50000
    lr_peak, lr_pct_warmup = 1e-3, 0.3
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
        callbacks=[Callback(log_plots, plot_interval)],
    )
    path = Path(f"results/gp/{exp}/{kernel}/{model_name}-seed-{cfg.seed}")
    path.parent.mkdir(parents=True, exist_ok=True)
    loss = validate(
        rng_valid, state, dataloader, valid_num_steps, path.with_suffix(".pkl")
    )
    wandb.log({"test_loss", loss})
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloader(exp: DictConfig, kernel: DictConfig):
    """Generates batches of GP samples."""
    gp = instantiate(kernel)
    s_dim = len(exp.s)
    s_grid = build_grid(exp.s).reshape(-1, s_dim)  # flatten spatial dims
    obs_noise, batch_size = exp.obs_noise, exp.batch_size
    valid_lens_test = jnp.repeat(exp.num_ctx.max + s_grid.shape[0], batch_size)
    min_s = jnp.array([axis["start"] for axis in exp.s])
    max_s = jnp.array([axis["stop"] for axis in exp.s])

    @jit
    def gen_s_random(rng: jax.Array):
        return random.uniform(rng, (exp.num_ctx.max, s_dim), minval=min_s, maxval=max_s)

    @jit
    def gen_batch(rng: jax.Array):
        rng_s_random, rng_valid_lens_ctx, rng_gp, rng_eps, rng = random.split(rng, 5)
        s_random = gen_s_random(rng_s_random)
        s = jnp.vstack([s_random, s_grid])
        var, ls, _z, f = gp.simulate(rng_gp, s, batch_size)
        valid_lens_ctx = random.randint(
            rng_valid_lens_ctx,
            (batch_size,),
            exp.num_ctx.min,
            exp.num_ctx.max,
        )
        s = jnp.repeat(s[None, ...], batch_size, axis=0)
        f_noisy = f + obs_noise * random.normal(rng_eps, f.shape)
        return s, f_noisy, valid_lens_ctx, s, f, valid_lens_test, var, ls

    def dataloader(rng: jax.Array):
        while True:
            rng_batch, rng = random.split(rng)
            yield gen_batch(rng_batch)

    return dataloader


def log_plots(
    step: int,
    rng_step: int,
    state: TrainState,
    batch: tuple,
    num_plots: int = 16,
):
    """Logs `num_plots` from the given batch."""
    rng_dropout, rng_extra = random.split(rng_step)
    s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, var, ls = batch
    f_mu, f_std, *_ = state.apply_fn(
        {"params": state.params, **state.kwargs},
        s_ctx,
        f_ctx,
        s_test,
        valid_lens_ctx,
        valid_lens_test,
        rngs={"dropout": rng_dropout, "extra": rng_extra},
    )
    if s_ctx.shape[-1] > 1:  # TODO(danj): implement 2D plot logging
        return
    paths = []
    for i in range(num_plots):
        v_ctx = valid_lens_ctx[i]
        s_ctx_i = s_ctx[i, :v_ctx].squeeze()
        f_ctx_i = f_ctx[i, :v_ctx].squeeze()
        v_test = valid_lens_test[i]
        s_test_i = s_test[i, :v_test].squeeze()
        f_test_i = f_test[i, :v_test].squeeze()
        f_mu_i = f_mu[i, :v_test].squeeze()
        f_std_i = f_std[i, :v_test].squeeze()
        if f_mu[i].shape != f_std[i].shape:  # marginal from tril cov
            f_std_i = jnp.diag(f_std[i]).squeeze()  # TODO(danj): is this valid?
        if f_mu.shape != f_test.shape:  # bootstrapped
            K = f_mu.shape[0] // f_test.shape[0]
            s = i * K
            f_mu_i = f_mu[s : s + K].squeeze()
            f_std_i = f_std[s : s + K].squeeze()
        path = plot_posterior_predictive(
            i, s_ctx_i, f_ctx_i, s_test_i, f_test_i, f_mu_i, f_std_i, var, ls
        )
        paths += [path]
    wandb.log({f"Step {step}": [wandb.Image(p) for p in paths]})


def plot_posterior_predictive(
    id: int,
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    s_test: jax.Array,
    f_test: jax.Array,
    f_mu: jax.Array,
    f_std: jax.Array,
    var: np.ndarray,
    ls: np.ndarray,
    hdi_prob=0.95,
):
    """Plots the posterior predictive alongside true values."""
    f_mu = f_mu[None, ...] if f_mu.ndim == 1 else f_mu
    f_std = f_std[None, ...] if f_std.ndim == 1 else f_std
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    f_lower, f_upper = f_mu - z_score * f_std, f_mu + z_score * f_std
    idx = jnp.argsort(s_test)
    plt.plot(s_test[idx], f_test[idx], color="black")
    plt.scatter(s_ctx, f_ctx, color="black")
    K = f_mu.shape[0]
    for i in range(K):
        f_mu_i = f_mu[i]
        plt.plot(s_test[idx], f_mu_i[idx], color="steelblue")
        f_lower_i, f_upper_i = f_lower[i], f_upper[i]
        plt.fill_between(
            s_test[idx],
            f_lower_i[idx],
            f_upper_i[idx],
            alpha=0.4 / K,
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


if __name__ == "__main__":
    main()
