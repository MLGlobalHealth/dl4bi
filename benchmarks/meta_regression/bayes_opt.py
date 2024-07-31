#!/usr/bin/env python3
from collections.abc import Callable
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb
from jax import jit, random
from jax.scipy import stats
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dsp.core import mask_from_valid_lens
from dsp.meta_regression.train_utils import (
    TrainState,
    build_gp_dataloader,
    cfg_to_run_name,
    load_ckpt,
    log_wandb_line,
    plot_posterior_predictive,
)


# NOTE: use the same configs as the Gaussian Process (GP) models
@hydra.main("configs/gp", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    path = f"results/gp/{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}/{run_name}"
    path = Path(path)
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=cfg.get("name", run_name),
        project="SPTx - Bayesian Optimization",
        reinit=True,  # allows reinitialization for multiple runs
    )
    cfg.data.batch_size = 1  # override GP batch argument
    print(OmegaConf.to_yaml(cfg))
    num_tasks, budget, num_init = 100, 50, 10
    rng = random.key(cfg.seed)
    rng_data, rng_opt = random.split(rng)
    dataloader = build_gp_dataloader(cfg.data, cfg.kernel)
    model_state, _ = load_ckpt(path.with_suffix(".ckpt"))
    model_fn = jit_model_fn(model_state)
    s_test, f_test = build_dataset(rng_data, dataloader, num_tasks)
    regret, s_ctx, f_ctx, f_mu, f_std = optimize(
        rng_opt, s_test, f_test, model_fn, num_init, budget
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    jnp.save(path.with_suffix(".npy"), regret)
    log_wandb_line(regret.mean(axis=0), "Regret mu")
    log_wandb_line(regret.std(axis=0), "Regret std")
    log_regret_dist(regret)
    log_worst_regret(
        s_ctx,
        f_ctx,
        s_test,
        f_test,
        f_mu,
        f_std,
        regret,
    )


def jit_model_fn(state: TrainState):
    @jit
    def model_fn(
        s_ctx: jax.Array,
        f_ctx: jax.Array,
        s_test: jax.Array,
        valid_lens_ctx: jax.Array,
        rng_extra: jax.Array,
    ):
        return state.apply_fn(
            {"params": state.params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            rngs={"extra": rng_extra},
        )

    return model_fn


def build_dataset(rng: jax.Array, dataloader: Callable, num_samples: int = 100):
    """Builds a dataset of `num_samples` with independent parameters.

    .. note::
        This is built sequentially because each "batch" from the dataloader
        requires an $O(n^3)$ calculation (for GPs). If we did this in parallel,
        some machines would run out of memory.
    """
    B = num_samples
    rng_data, rng_permute = random.split(rng)
    batches = dataloader(rng_data)
    s_tests, f_tests = [], []
    for i in tqdm(range(B), desc="Building dataset"):
        _, _, _, s_test, f_test, *_ = next(batches)
        s_tests += [s_test]
        f_tests += [f_test]
    s_test, f_test = jnp.vstack(s_tests), jnp.vstack(f_tests)
    s_test = random.permutation(rng_permute, s_test, axis=1, independent=True)
    f_test = random.permutation(rng_permute, f_test, axis=1, independent=True)
    D_s, D_f = s_test.shape[-1], f_test.shape[-1]
    return s_test.reshape(B, -1, D_s), f_test.reshape(B, -1, D_f)  # [B, L, D]


def optimize(
    rng: jax.Array,
    s_test: jax.Array,  # [B, L, D_s]
    f_test: jax.Array,  # [B, L, D_f]
    model_fn: Callable,
    num_init: int = 1,
    budget: int = 50,
):
    (B, L_test, D_s), L_ctx = s_test.shape, num_init + budget
    # NOTE: This only works with 1 dimensional D_f because you have to select
    # 1 position index to add to the context set on each iteration, and if
    # different positions optimize different elements of the output vector f,
    # you'd have to arbitrarily select one of them.
    assert f_test.shape[-1] == 1, "Can only optimize functions with a single output!"
    s_ctx = jnp.zeros((B, L_ctx, D_s))
    f_ctx = jnp.zeros((B, L_ctx, 1))
    s_ctx = s_ctx.at[:, :num_init, :].set(s_test[:, :num_init, :])
    f_ctx = f_ctx.at[:, :num_init, :].set(f_test[:, :num_init, :])
    valid_lens_ctx = jnp.repeat(num_init, B)
    mask = mask_from_valid_lens(L_ctx, valid_lens_ctx)
    opt_min = jnp.full((B, budget + 1, 1), jnp.inf)
    opt_min = opt_min.at[:, 0, :].set(f_ctx.min(axis=1, where=mask, initial=jnp.inf))
    B_idx = jnp.arange(B)
    for i in tqdm(range(budget), desc="Optimizing"):
        rng_extra, rng = random.split(rng)
        f_mu, f_std, *_ = model_fn(s_ctx, f_ctx, s_test, valid_lens_ctx, rng_extra)
        if f_mu.shape != f_test.shape:  # bootstrapped
            K = f_mu.shape[0] // f_test.shape[0]
            f_mu = f_mu.reshape(B, K, L_test, 1)
            f_std = f_std.reshape(B, K, L_test, 1)
            # law of total variance
            f_var = (f_std**2).mean(1) + (f_mu**2).mean(1) - (f_mu.mean(1)) ** 2
            f_std = jnp.sqrt(f_var)
            f_mu = f_mu.mean(axis=1)
        ei = expected_improvement(opt_min[:, [i], :], f_mu, f_std)
        min_idx = jnp.argmax(ei, axis=1).squeeze()  # [B]
        s_ctx = s_ctx.at[B_idx, num_init + i, :].set(s_test[B_idx, min_idx, :])
        f_ctx = f_ctx.at[B_idx, num_init + i, :].set(f_test[B_idx, min_idx, :])
        valid_lens_ctx += 1
        mask = mask_from_valid_lens(L_ctx, valid_lens_ctx)
        opt_min = opt_min.at[:, i + 1, :].set(
            f_ctx.min(axis=1, where=mask, initial=jnp.inf)
        )
    global_min = f_test.min(axis=1, keepdims=True)
    regret = opt_min - global_min
    return regret.squeeze(), s_ctx, f_ctx, f_mu, f_std


@jit
def expected_improvement(
    f_min: jax.Array,
    f_mu: jax.Array,
    f_std: jax.Array,
    f_std_jitter: float = 1e-5,  # Copied from BayesO constants
):
    """A JAX implementation of BayesO's `acquisition.ei`."""
    d = f_min - f_mu
    z = d / (f_std + f_std_jitter)
    return d * stats.norm.cdf(z) + f_std * stats.norm.pdf(z)


def log_regret_dist(regret: jax.Array, hdi_prob: float = 0.95):
    regret = regret[:, 1:]  # ignore iteration 0, before model selected anything
    mu, std = regret.mean(axis=0), regret.std(axis=0)
    z = jnp.abs(stats.norm.ppf((1 - hdi_prob) / 2))
    iter = jnp.arange(regret.shape[1])
    plt.plot(iter, mu, color="black")
    plt.fill_between(
        iter,
        jnp.clip(mu - z * std, min=0),  # regret can't be negative
        mu + z * std,
        color="steelblue",
        alpha=0.4,
        interpolate=True,
    )
    ax = plt.gca()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Minimum Simple Regret")
    path = "/tmp/regret_dist.png"
    plt.title("Regret Distribution")
    plt.savefig(path, dpi=125)
    plt.clf()
    wandb.log({"Regret Distribution": wandb.Image(path)})
    return path


def log_worst_regret(
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    s_test: jax.Array,
    f_test: jax.Array,
    f_mu: jax.Array,
    f_std: jax.Array,
    regret: jax.Array,
    num_samples: int = 16,
    hdi_prob: float = 0.95,
):
    if s_test.shape[-1] != 1:  # TODO(danj): support 2D
        return
    worst_idxs = jnp.argsort(regret[:, -1], descending=True)[:num_samples]
    paths = []
    for rank, i in enumerate(worst_idxs):
        # plot posterior predictive
        s_ctx_i, f_ctx_i = s_ctx[i].squeeze(), f_ctx[i].squeeze()
        s_test_i, f_test_i = s_test[i].squeeze(), f_test[i].squeeze()
        f_mu_i, f_std_i = f_mu[i].squeeze(), f_std[i].squeeze()
        if f_mu[i].shape != f_std[i].shape:  # marginal from tril cov
            f_std_i = jnp.diag(f_std[i]).squeeze()  # TODO(danj): valid?
        if f_mu.shape != f_test.shape:  # bootstrapped
            K = f_mu.shape[0] // f_test.shape[0]
            s = i * K
            f_mu_i = f_mu[s : s + K].squeeze()
            f_std_i = f_std[s : s + K].squeeze()
        fig = plot_posterior_predictive(
            s_ctx_i, f_ctx_i, s_test_i, f_test_i, f_mu_i, f_std_i
        )
        ax = fig.axes[0]
        # add estimated min and true global min
        opt_min_idx = f_ctx_i.argmin()
        global_min_idx = f_test_i.argmin()
        ax.plot(s_ctx_i[opt_min_idx], f_ctx_i[opt_min_idx], "ro", alpha=0.5)
        ax.plot(s_test_i[global_min_idx], f_test_i[global_min_idx], "go", alpha=0.5)
        paths += [f"/tmp/worst_regret_{rank+1}_sample_{i}.png"]
        fig.suptitle(f"Sample {i}, Regret {regret[i, -1]:.3f}")
        fig.savefig(paths[-1], dpi=125)
        plt.clf()
        plt.close(fig)
    wandb.log({"Worst Regrets": [wandb.Image(p) for p in paths]})


if __name__ == "__main__":
    main()
