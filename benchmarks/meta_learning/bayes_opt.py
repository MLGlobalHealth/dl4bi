#!/usr/bin/env python3
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb
from gp import build_dataloader
from jax import jit, random
from jax.scipy import stats
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dl4bi.core.train import TrainState, load_ckpt
from dl4bi.core.utils import mask_from_valid_lens
from dl4bi.meta_learning.data.spatial import SpatialBatch
from dl4bi.meta_learning.utils import cfg_to_run_name


# NOTE: use the same configs as the Gaussian Process (GP) models
@hydra.main("configs/gp", config_name="default", version_base=None)
def main(cfg: DictConfig):
    project_parent = cfg.get("project_parent")
    if re.match(".*Gaussian Process.*", cfg.project, re.IGNORECASE):
        project_parent = project_parent or cfg.project
        cfg.project = "Bayesian Optimization"
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    kernel = cfg.kernel._target_.split(".")[-1]
    path = f"results/{project_parent}/{cfg.data.name}/{kernel}/{cfg.seed}/{run_name}"
    path = Path(path)
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    cfg.data.batch_size = 1  # override GP batch argument
    print(OmegaConf.to_yaml(cfg))
    num_tasks, budget, num_init = 100, 50, 10
    rng = random.key(cfg.seed)
    rng_data, rng_opt = random.split(rng)
    dataloader = build_dataloader(cfg.data, cfg.kernel)
    model_state, _ = load_ckpt(path.with_suffix(".ckpt"))
    model_fn = jit_model_fn(model_state)
    s_test, f_test = build_dataset(rng_data, dataloader, num_tasks)
    regret, s_ctx, f_ctx, f_mu, f_std = optimize(
        rng_opt, s_test, f_test, model_fn, num_init, budget
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    jnp.save(path.with_suffix(".npy"), regret)
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
        mask_ctx: jax.Array,
        rng_extra: jax.Array,
    ):
        return state.apply_fn(
            {"params": state.params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            mask_ctx,
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
    batches = dataloader(rng)
    s_tests, f_tests = [], []
    for i in tqdm(range(num_samples), desc="Building dataset"):
        b = next(batches)
        s_tests += [b.s_test]
        f_tests += [b.f_test]
    return jnp.vstack(s_tests), jnp.vstack(f_tests)


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
    est_min = jnp.full((B, budget + 1, 1), jnp.inf)
    est_min = est_min.at[:, 0, :].set(
        f_ctx.min(axis=1, where=mask[..., None], initial=jnp.inf)
    )
    B_idx = jnp.arange(B)
    for i in tqdm(range(budget), desc="Optimizing"):
        rng_extra, rng = random.split(rng)
        output = model_fn(s_ctx, f_ctx, s_test, mask, rng_extra)
        if isinstance(output, tuple):  # latent
            output, _ = output  # throw away latent samples
        f_mu, f_std = output.mu, output.std
        ei = expected_improvement(est_min[:, [i], :], f_mu, f_std)
        min_idx = jnp.argmax(ei, axis=1).squeeze()  # [B]
        s_ctx = s_ctx.at[B_idx, num_init + i, :].set(s_test[B_idx, min_idx, :])
        f_ctx = f_ctx.at[B_idx, num_init + i, :].set(f_test[B_idx, min_idx, :])
        valid_lens_ctx += 1
        mask = mask_from_valid_lens(L_ctx, valid_lens_ctx)
        est_min = est_min.at[:, i + 1, :].set(
            f_ctx.min(axis=1, where=mask[..., None], initial=jnp.inf)
        )
    opt_min = f_test.min(axis=1, keepdims=True)
    regret = est_min - opt_min
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


@jit
def upper_confidence_bound(
    f_min: jax.Array,
    f_mu: jax.Array,
    f_std: jax.Array,
    num_samples: int,
    kappa: float = 2.0,
):
    kappa *= jnp.log(num_samples + 1)  # scale as training set increases
    return -f_mu + f_std * kappa * jnp.log(num_samples + 1)


@jit
def probability_of_improvement(
    f_min: jax.Array,
    f_mu: jax.Array,
    f_std: jax.Array,
    f_std_jitter: float = 1e-5,
):
    z = (f_min - f_mu) / (f_std + f_std_jitter)
    return stats.norm.cdf(z)


def log_worst_regret(
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    s_test: jax.Array,
    f_test: jax.Array,
    f_mu: jax.Array,
    f_std: jax.Array,
    regret: jax.Array,
    num_samples: int = 16,
):
    worst_idxs = jnp.argsort(regret[:, -1], descending=True)[:num_samples]
    L_ctx, L_test = s_ctx.shape[1], s_test.shape[1]
    b = SpatialBatch(
        None,
        s_ctx[worst_idxs],
        f_ctx[worst_idxs],
        mask_from_valid_lens(L_ctx, jnp.repeat(L_ctx, num_samples)),
        None,
        s_test[worst_idxs],
        f_test[worst_idxs],
        mask_from_valid_lens(L_test, jnp.repeat(L_test, num_samples)),
        jnp.arange(L_test),
        s_shape=s_test.shape,
    )
    fig = b.plot_1d(f_mu[worst_idxs], f_std[worst_idxs], subtitle="with worst regret")
    regret = regret[:, -1][worst_idxs]
    for i, ax in enumerate(fig.axes):
        est_min = b.f_ctx[i].argmin()
        opt_min = b.f_test[i].argmin()
        x, y = b.s_ctx[i, est_min, 0], b.f_ctx[i, est_min, 0]
        ax.plot(x, y, "ro")
        ax.text(
            0.975,
            0.95,
            f"Regret: {regret[i]:0.2f}",
            transform=ax.transAxes,  # Use axes-relative coordinates
            fontsize=16,
            verticalalignment="top",
            horizontalalignment="right",
        )
        ax.plot(b.s_test[i, opt_min, 0], b.f_test[i, opt_min, 0], "go")
    path = f"/tmp/bayes_opt_{datetime.now().isoformat()}.png"
    fig.savefig(path)
    plt.close(fig)
    wandb.log({"Worst Regrets": wandb.Image(path)})


if __name__ == "__main__":
    main()
