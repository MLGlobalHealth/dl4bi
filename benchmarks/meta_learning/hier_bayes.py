#!/usr/bin/env python
from pathlib import Path
from typing import Optional

import hydra
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import optax
import wandb
from jax import jit, random
from numpyro.infer import MCMC, NUTS, Predictive
from omegaconf import DictConfig, OmegaConf
from sps.gp import GP
from sps.kernels import rbf

from dl4bi.core import dist_spatial
from dl4bi.meta_learning import TNPKR
from dl4bi.meta_learning.train_utils import (
    cfg_to_run_name,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    save_ckpt,
    select_steps,
    train,
)

# TODO:
# 1. Actually create batches from data loaders, s_ctx, valid_lens, etc
# 2. Add configs
# 3. Add inference

# mcmc = infer(rng_mcmc, args, numpyro_spatial_model, s, f)
# mcmc.print_summary()
# posterior_samples = mcmc.get_samples()
# ll = log_likelihood(numpryo_spatial_model, posterior_samples, s=s, f=f)
# print(ll)


@hydra.main("configs/hier_bayes", config_name="default", version_base=None)
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
    dataloader = build_jax_dataloader(jax_spatial_model, 10, 64)
    lr_schedule = cosine_annealing_lr(
        cfg.train_num_steps,
        cfg.lr_peak,
        cfg.lr_pct_warmup,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.clip_max_norm),
        optax.yogi(lr_schedule),
    )
    # model = instantiate(cfg.model)
    model = TNPKR(dist=lambda q, r: dist_spatial(q[..., [0]], r[..., [0]]))
    train_step, valid_step = select_steps(model)
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
    )
    metrics = evaluate(
        rng_test,
        state,
        valid_step,
        dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = f"results/{cfg.project}/{cfg.seed}/{run_name}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_numpyro_dataloader(
    model,
    s: jax.Array,
    num_features: int,
    batch_size: int = 64,
):
    """Same s, different x per batch over fixed effect and spatial priors."""
    s = jnp.linspace(-2, 2, 128)
    B, S, D = batch_size, s.shape[0], num_features  # batch_size ~= number of timesteps
    prior_pred = jit(Predictive(model, num_samples=B))
    s_batch = jnp.repeat(s[None, :, None], B, axis=0)
    stack = lambda *args: jnp.concatenate(args, axis=-1)

    def dataloader(rng: jax.Array):
        while True:
            rng_x, rng_f, rng = random.split(rng, 3)
            # could replace with VAE to generate x samples
            x = random.normal(rng_x, (S, D))
            x_batch = jnp.repeat(x[None, ...], B, axis=0)
            samples = prior_pred(rng_f, x=x, s=s)
            yield stack(s_batch, x_batch), samples["f"]

    return dataloader


def numpyro_spatial_model(
    x: jax.Array,
    s: jax.Array,
    f: Optional[jax.Array] = None,
    jitter: float = 1e-5,
):
    """Generic Spatiotemporal model with random spatial effects.

    Args:
        s: Array of input locations, `[S]`.
        x: Array of input covariates, `[S, D]`.
        f: Observed function values, `[S, 1]`.
    """

    S, D = x.shape
    ls = numpyro.sample("ls", dist.Beta(3, 7))
    k = rbf(s, s, var=1.0, ls=ls) + jitter * jnp.eye(S)
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(D), jnp.ones(D)))
    f_mu_x = x @ beta
    f_mu_s = numpyro.sample("f_mu_s", dist.MultivariateNormal(jnp.zeros(S), k))
    f_sigma = numpyro.sample("f_sigma", dist.HalfNormal(0.1))
    numpyro.sample("f", dist.Normal(f_mu_x + f_mu_s, f_sigma))


def build_jax_dataloader(model, num_features: int, batch_size: int = 64):
    """Same s, different x per batch over fixed effect and spatial priors."""
    s = jnp.linspace(-2, 2, 128)
    B, S, D = batch_size, s.shape[0], num_features  # batch_size ~= number of timesteps
    s_batch = jnp.repeat(s[None, :, None], B, axis=0)
    stack = lambda *args: jnp.concatenate(args, axis=-1)

    def dataloader(rng: jax.Array):
        while True:
            rng_x, rng_f, rng = random.split(rng, 3)
            x = random.normal(rng_x, (S, D))
            x_batch = jnp.repeat(x[None, ...], B, axis=0)
            f = model(rng_f, x, s, B)
            yield stack(s_batch, x_batch), f

    return dataloader


def jax_spatial_model(
    rng: jax.Array,
    x: jax.Array,
    s: jax.Array,
    batch_size: int = 64,
):
    """A much faster, pure JAX spatiotemporal model.

    Technically this isn't the same model since the GP class samples
    the GP priors once per batch in order to amortize the cost of
    the Cholesky decomposition.
    """
    rng_gp, rng = random.split(rng)
    f_mu_s, *_ = GP(rbf).simulate(rng_gp, s, batch_size)  # can't jit this
    f_mu_s = f_mu_s.squeeze()  # [B, S, 1] -> [B, S]
    return _non_gp_jax_spatial_model(rng, x, f_mu_s)


@jit
def _non_gp_jax_spatial_model(rng: jax.Array, x: jax.Array, f_mu_s: jax.Array):
    (S, D), B = x.shape, f_mu_s.shape[0]
    rng_beta, rng_sigma, rng_noise = random.split(rng, 3)
    beta = random.normal(rng_beta, (B, D))
    f_mu_x = beta @ x.T  # [B, S]
    f_sigma = 0.1 * jnp.abs(random.normal(rng_sigma, (B,)))
    f = f_mu_x + f_mu_s + f_sigma[:, None] * random.normal(rng_noise, (B, S))
    return f


def infer(rng, args, model, s, f):
    sampler = NUTS(model)
    mcmc = MCMC(
        sampler,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
    )
    mcmc.run(rng, s, f)
    return mcmc


if __name__ == "__main__":
    main()
