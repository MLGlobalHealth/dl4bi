#!/usr/bin/env python3
from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax
import wandb
from jax import jit, random
from jax.scipy.stats import norm
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid

from dl4bi.meta_learning.train_utils import (
    Callback,
    TrainState,
    cfg_to_run_name,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    log_img_plots,
    save_ckpt,
    select_steps,
    train,
)


@hydra.main("configs/gp", config_name="default", version_base=None)
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
    dataloader = build_gp_dataloader(cfg.data, cfg.kernel)
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
    clbk = log_posterior_predictive_plots
    if cfg.data.name == "2d":
        H, W = cfg.data.s[0].num, cfg.data.s[1].num
        clbk = partial(
            log_2d_grid_gp_plots,
            shape=(H, W, 1),
            data=cfg.data,
            kernel=cfg.kernel,
        )
    train_step, valid_step = select_steps(
        model, consistency_loss=cfg.get("consistency_loss")
    )
    state = train(
        rng_train,
        model,
        optimizer,
        train_step,
        valid_step,
        dataloader,
        dataloader,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[Callback(clbk, cfg.plot_interval)],
    )
    metrics = evaluate(
        rng_test,
        state,
        valid_step,
        dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = f"results/{cfg.project}/{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}/{run_name}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_gp_dataloader(data: DictConfig, kernel: DictConfig):
    """Generates batches of GP samples."""
    gp = instantiate(kernel)
    B, S = data.batch_size, len(data.s)
    Nc_min, Nc_max = data.num_ctx.min, data.num_ctx.max
    s_g = build_grid(data.s).reshape(-1, S)  # flatten spatial dims
    L = Nc_max + s_g.shape[0]  # L = num test or all points
    obs_noise, B = data.obs_noise, data.batch_size
    valid_lens_test = jnp.repeat(L, B)
    s_min = jnp.array([axis["start"] for axis in data.s])
    s_max = jnp.array([axis["stop"] for axis in data.s])
    batchify = jit(lambda x: jnp.repeat(x[None, ...], B, axis=0))

    def gen_batch(rng: jax.Array):
        rng_s, rng_gp, rng_v, rng_ctx, rng_obs = random.split(rng, 5)
        s_r = random.uniform(rng_s, (Nc_max, S), jnp.float32, s_min, s_max)
        s = jnp.vstack([s_r, s_g])
        f, var, ls, period, *_ = gp.simulate(rng_gp, s, B)
        valid_lens_ctx = random.randint(rng_v, (B,), Nc_min, Nc_max)
        s = batchify(s)
        s_ctx = s[:, :Nc_max, :]
        f_ctx = f + obs_noise * random.normal(rng_ctx, f.shape)
        f_ctx = f_ctx[:, :Nc_max, :]
        if data.get("noisy_test", False):
            f = f + obs_noise * random.normal(rng_obs, f.shape)
        return s_ctx, f_ctx, valid_lens_ctx, s, f, valid_lens_test, var, ls, period

    def dataloader(rng: jax.Array):
        while True:
            rng_i, rng = random.split(rng)
            yield gen_batch(rng_i)

    return dataloader


def build_2d_grid_gp_dataloader(data: DictConfig, kernel: DictConfig):
    """A custom 2D GP dataloader in which generated context and test points
        reside only on the 2d grid.

    .. note::
        The dataloader used for training and testing uses context points
        on a continuous domain, while this only uses points on a grid for
        visualization purposes.
    """
    gp = instantiate(kernel)
    B, S, obs_noise = data.batch_size, len(data.s), data.obs_noise
    Nc_min, Nc_max = data.num_ctx.min, data.num_ctx.max
    s_g = build_grid(data.s).reshape(-1, S)  # flatten spatial dims
    batchify = jit(lambda x: jnp.repeat(x[None, ...], B, axis=0))
    s = batchify(s_g)
    L = s.shape[1]
    valid_lens_test = jnp.repeat(L, B)

    def gen_batch(rng: jax.Array):
        rng_gp, rng_eps, rng_v, rng_permute, rng = random.split(rng, 5)
        f, var, ls, period, *_ = gp.simulate(rng_gp, s_g, B)
        f_noisy = f + obs_noise * random.normal(rng_eps, f.shape)
        valid_lens_ctx = random.randint(rng_v, (B,), Nc_min, Nc_max)
        permute_idx = random.choice(rng_permute, L, (L,), replace=False)
        inv_permute_idx = jnp.argsort(permute_idx)
        s_permuted = s[:, permute_idx, :]
        f_permuted = f[:, permute_idx, :]
        f_noisy_permuted = f_noisy[:, permute_idx, :]
        s_ctx = s_permuted[:, :Nc_max, :]
        f_ctx = f_noisy_permuted[:, :Nc_max, :]
        return (
            s_ctx,
            f_ctx,
            valid_lens_ctx,
            s_permuted,  # s_test
            f_permuted,  # f_test
            valid_lens_test,
            s,  # add full original for plotting
            f,
            inv_permute_idx,
        )

    def dataloader(rng: jax.Array):
        while True:
            rng_i, rng = random.split(rng)
            yield gen_batch(rng_i)

    return dataloader


def log_posterior_predictive_plots(
    step: int,
    rng_step: int,
    state: TrainState,
    batch: tuple,
    num_plots: int = 16,
):
    rng_dropout, rng_extra = random.split(rng_step)
    s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, var, ls, period = (
        batch
    )
    output = state.apply_fn(
        {"params": state.params, **state.kwargs},
        s_ctx,
        f_ctx,
        s_test,
        valid_lens_ctx,
        valid_lens_test,
        rngs={"dropout": rng_dropout, "extra": rng_extra},
    )
    if isinstance(output[1], tuple):  # latent or bootstrapped
        output, _ = output  # throw away latent / base samples
    f_mu, f_std = output
    paths = plot_posterior_predictives(
        s_ctx,
        f_ctx,
        valid_lens_ctx,
        s_test,
        f_test,
        valid_lens_test,
        var,
        ls,
        period,
        f_mu,
        f_std,
        num_plots,
    )
    wandb.log({f"Step {step}": [wandb.Image(p) for p in paths]})


def log_2d_grid_gp_plots(
    step: int,
    rng_step: int,
    state: TrainState,
    batch: tuple,
    shape: tuple[int, int, int],
    data: DictConfig,
    kernel: DictConfig,
    num_plots: int = 16,
):
    """Logs `num_plots` from the given batch for 2D GPs."""
    rng_step, rng_batch = random.split(rng_step)
    cmap = mpl.colormaps.get_cmap("Spectral_r")
    cmap.set_bad("grey")
    batch = next(build_2d_grid_gp_dataloader(data, kernel)(rng_batch))
    log_img_plots(step, rng_step, state, batch, shape, cmap=cmap, num_plots=num_plots)


def plot_posterior_predictives(
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    valid_lens_ctx: jax.Array,
    s_test: jax.Array,
    f_test: jax.Array,
    valid_lens_test: jax.Array,
    var: jax.Array,
    ls: jax.Array,
    period: jax.Array | None,
    f_mu: jax.Array,
    f_std: jax.Array,
    num_plots: int = 16,
):
    """Plots `num_plots` from the given batch."""
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
        title = f"Sample {i} (var: {var[i]:0.2f}, ls: {ls[i]:0.2f}"
        title += (
            f", period: {period[i]:0.2f})" if period and jnp.isfinite(period) else ")"
        )
        fig = plot_posterior_predictive(
            s_ctx_i, f_ctx_i, s_test_i, f_test_i, f_mu_i, f_std_i
        )
        fig.suptitle(title)
        paths += [f"/tmp/{datetime.now().isoformat()} - {title}.png"]
        fig.savefig(paths[-1], dpi=125)
        plt.clf()
        plt.close(fig)
    return paths


def plot_posterior_predictive(
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    s_test: jax.Array,
    f_test: jax.Array,
    f_mu: jax.Array,
    f_std: jax.Array,
    hdi_prob: float = 0.95,
):
    """Plots the posterior predictive alongside true values."""
    f_mu = f_mu[None, ...] if f_mu.ndim == 1 else f_mu
    f_std = f_std[None, ...] if f_std.ndim == 1 else f_std
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    f_lower, f_upper = f_mu - z_score * f_std, f_mu + z_score * f_std
    idx = jnp.argsort(s_test)
    plt.plot(s_test[idx], f_test[idx], color="black")
    plt.scatter(s_ctx, f_ctx, color="black", alpha=0.75)
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
    return plt.gcf()


if __name__ == "__main__":
    main()
