#!/usr/bin/env python3
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import hydra
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import wandb
from hydra.utils import instantiate
from jax import jit, random, vmap
from numpyro.infer import MCMC, NUTS, Predictive
from omegaconf import DictConfig, OmegaConf
from sps.kernels import rbf
from sps.utils import build_grid

from dl4bi.core.train import (
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.spatial import SpatialData
from dl4bi.meta_learning.utils import cfg_to_run_name


@hydra.main("configs/generic_spatiotemporal", config_name="default", version_base=None)
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
    dataloader = build_dataloader(cfg.data)
    optimizer = instantiate(cfg.optimizer)
    model = instantiate(cfg.model)
    output_fn = model.output_fn
    model = model.copy(output_fn=lambda x: output_fn(x, min_std=0.05))
    state = train(
        rng_train,
        model,
        optimizer,
        model.train_step,
        cfg.train_num_steps,
        dataloader,
        model.valid_step,
        cfg.valid_interval,
        cfg.valid_num_steps,
        dataloader,
    )
    metrics = evaluate(
        rng_test,
        state,
        model.valid_step,
        dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloader(data: DictConfig):
    B, D_x, D_s = data.batch_size, data.num_features, len(data.s)
    s_grid = build_grid(data.s)
    s_flat = s_grid.reshape(-1, D_s)
    s_batch = jnp.repeat(s_grid[None, ...], B, axis=0)
    prior_pred = jit(Predictive(generic_spatiotemporal_model, num_samples=B))

    def dataloader(rng: jax.Array):
        L = s_flat.shape[0]
        while True:
            # reuse x for several batches to avoid too much
            # noise in early training
            rng_x, rng = random.split(rng)
            x = random.normal(rng_x, (L, D_x))
            x_batch = jnp.repeat(x[None, ...], B, axis=0)
            for i in range(data.num_x_batches):
                rng_p, rng_b, rng = random.split(rng, 3)
                samples = prior_pred(rng_p, x, s_flat)
                f = samples["f"][..., None]  # [B, L, 1]
                yield SpatialData(x_batch, s_batch, f).batch(
                    rng_b,
                    data.num_ctx_min,
                    data.num_ctx_max,
                    data.num_test,
                )

    return dataloader


def generic_spatiotemporal_model(
    x: jax.Array,  # [L, D_x]
    s: jax.Array,  # [L, D_s]
    f: Optional[jax.Array] = None,  # [L, 1]
    jitter: float = 1e-5,
    **kwargs,
):
    """Generic Spatiotemporal model with fixed and random spatial effects.

    Args:
        x: Array of input covariates, `[L, D]`.
        s: Array of input locations, `[L, S]`.
        f: Observed function values, `[L, 1]`.
    """

    L, D = x.shape
    var = numpyro.deterministic("var", 1.0)
    ls = numpyro.sample("ls", dist.Beta(3, 7))
    k = rbf(s, s, var, ls) + jitter * jnp.eye(L)
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(D), jnp.ones(D)))
    f_mu_x = x @ beta
    f_mu_s = numpyro.sample("f_mu_s", dist.MultivariateNormal(0, k))
    f_mu = f_mu_x + f_mu_s
    f_obs_noise = numpyro.sample("f_obs_noise", dist.HalfNormal(0.1))
    numpyro.sample("f", dist.Normal(f_mu, f_obs_noise), obs=f)


@partial(jit, static_argnames=("jitter",))
def pointwise_post_pred(
    rng: jax.Array,
    s_ctx: jax.Array,  # [L_ctx, S]
    s_test: jax.Array,  # [L_test, S]
    x_test: jax.Array,  # [L_test, D]
    beta: jax.Array,  # [N, L_test, D]
    var: jax.Array,  # [N]
    ls: jax.Array,  # [N]
    f_mu_s: jax.Array,  # [N, L_ctx]
    f_obs_noise: jax.Array,  # [N]
    jitter: float = 1e-5,
    **kwargs,
):
    """Calculates the pointwise posterior predictive."""
    N = ls.shape[0]
    f = vmap(post_pred, in_axes=(0, None, None, None, 0, 0, 0, 0, 0, None))(
        random.split(rng, N),
        s_ctx,
        s_test,
        x_test,
        beta,
        var,
        ls,
        f_mu_s,
        f_obs_noise,
        jitter,
    )  # f: [N, L]
    return f.mean(axis=0), f.std(axis=0)


@partial(jit, static_argnames=("jitter",))
def post_pred(
    rng: jax.Array,
    s_ctx: jax.Array,  # [L_ctx, D_s]
    s_test: jax.Array,  # [L_test, D_s]
    x_test: jax.Array,  # [L_test, D_x]
    beta: jax.Array,  # [D]
    var: jax.Array,  # [1]
    ls: jax.Array,  # [1]
    f_mu_s: jax.Array,  # [L_ctx, 1]
    f_obs_noise: jax.Array,  # [1]
    jitter: float = 1e-5,
):
    """Generates a posterior predictive sample.

    Source: https://num.pyro.ai/en/stable/examples/gp.html
    """
    rng_gp, rng_noise = random.split(rng)
    f_mu_s = condition_gp(rng_gp, s_ctx, f_mu_s, s_test, var, ls, rbf, jitter)
    f_mu_x = beta @ x_test.T
    f_mu = f_mu_s + f_mu_x
    f = f_mu + f_obs_noise * random.normal(rng_noise, f_mu.shape)
    return f


@partial(jit, static_argnames=("jitter", "kernel"))
def condition_gp(
    rng: jax.Array,
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    s_test: jax.Array,
    var: jax.Array,
    ls: jax.Array,
    kernel: Callable = rbf,
    jitter=1e-5,
):
    """Conditions a GP on observed points.

    Source: https://num.pyro.ai/en/stable/examples/gp.html
    """
    L_ctx, L_test = s_ctx.shape[0], s_test.shape[0]
    k_tt = kernel(s_test, s_test, var, ls) + jitter * jnp.eye(L_test)
    k_tc = kernel(s_test, s_ctx, var, ls)
    k_cc = kernel(s_ctx, s_ctx, var, ls) + jitter * jnp.eye(L_ctx)
    K_cc_cho = jax.scipy.linalg.cho_factor(k_cc)
    K = k_tt - k_tc @ jax.scipy.linalg.cho_solve(K_cc_cho, k_tc.T)
    mu = k_tc @ jax.scipy.linalg.cho_solve(K_cc_cho, f_ctx)
    noise = jnp.sqrt(jnp.clip(jnp.diag(K), 0.0)) * random.normal(rng, L_test)
    return mu + noise


def run_hmc(rng: jax.Array, model: Callable, infer: DictConfig, **kwargs):
    mcmc = MCMC(
        NUTS(model),
        num_warmup=infer.num_warmup,
        num_samples=infer.num_samples,
        num_chains=infer.num_chains,
    )
    mcmc.run(rng, **kwargs)
    return mcmc


if __name__ == "__main__":
    main()
