#!/usr/bin/env python
from pathlib import Path
from typing import Callable, Optional

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
from sps.utils import build_grid

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
# 1. Add inference
# 2. Add plots
# 3. Can you use distance bias on covariates too??

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
    dataloader = build_dataloader(jax_spatial_prior_pred_f, cfg.data)
    batches = dataloader(rng)
    # from tqdm import tqdm
    # import sys

    # for _ in tqdm(range(1000)):
    #     b = next(batches)

    # sys.exit(0)
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


def build_dataloader(prior_pred: Callable, data: DictConfig):
    """Generates batches of model samples.

    Args:
        prior_pred: Callable of (x: [L, D], s: [L, S], batch_size) -> f: [B, L, 1]
    """
    B, S, D = data.batch_size, len(data.s), data.num_features
    Nc_min, Nc_max = data.num_ctx.min, data.num_ctx.max
    s_g = build_grid(data.s).reshape(-1, S)  # flatten spatial dims
    L = Nc_max + s_g.shape[0]  # L = num_test or all points
    valid_lens_test = jnp.repeat(L, B)
    s_min = jnp.array([axis["start"] for axis in data.s])
    s_max = jnp.array([axis["stop"] for axis in data.s])
    batchify = lambda x: jnp.repeat(x[None, ...], B, axis=0)

    def gen_batch(rng: jax.Array):
        rng_s, rng_x, rng_v = random.split(rng, 3)
        x = random.normal(rng_x, (L, D))  # features for every location
        s_r = random.uniform(rng_s, (Nc_max, S), jnp.float32, s_min, s_max)
        s = jnp.vstack([s_r, s_g])
        f = prior_pred(rng, x, s, B)  # x: [L, D], s: [L, S], f: [B, L, 1]
        x, s = batchify(x), batchify(s)  # x: [B, L, D], s: [B, L, S]
        valid_lens_ctx = random.randint(rng_v, (B,), jnp.float32, Nc_min, Nc_max)
        s = jnp.concatenate([s, x], axis=-1)  # [B, L, S + D]
        s_ctx = s[:, :Nc_max, :]
        f_ctx = f[:, :Nc_max, :]
        return s_ctx, f_ctx, valid_lens_ctx, s, f, valid_lens_test

    def dataloader(rng: jax.Array):
        while True:
            rng_i, rng = random.split(rng)
            yield gen_batch(rng_i)

    return dataloader


def jax_spatial_prior_pred_f(
    rng: jax.Array,
    x: jax.Array,  # [L, D]
    s: jax.Array,  # [L, S]
    batch_size: int = 64,
):
    """A faster, pure JAX spatiotemporal model.

    Technically this isn't the same model since the GP class samples
    the GP priors once per batch in order to amortize the cost of
    the Cholesky decomposition.
    """
    rng_gp, rng_rest = random.split(rng)
    # NOTE: can't jit this; hlo lowering fails on cholesky?
    f_mu_s, *_ = GP(rbf).simulate(rng_gp, s, batch_size)
    f = _jax_spatial_prior_pred_rest(rng_rest, x, f_mu_s.squeeze())
    return f[..., None]  # [B, L, 1]


@jit
def _jax_spatial_prior_pred_rest(rng: jax.Array, x: jax.Array, f_mu_s: jax.Array):
    B, (L, D) = f_mu_s.shape[0], x.shape
    rng_beta, rng_sigma, rng_noise = random.split(rng, 3)
    beta = random.normal(rng_beta, (B, D))
    f_mu_x = beta @ x.T  # [B, L]
    f_sigma = 0.1 * jnp.abs(random.normal(rng_sigma, (B,)))
    f = f_mu_x + f_mu_s + f_sigma[:, None] * random.normal(rng_noise, (B, L))
    return f


@jit
def numpyro_spatial_prior_pred_f(
    rng: jax.Array,
    x: jax.Array,  # [L, D]
    s: jax.Array,  # [L, S]
    batch_size: int = 64,
):
    B = batch_size
    prior_pred = Predictive(numpyro_spatial_model, num_samples=B)
    samples = prior_pred(rng, x=x, s=s)
    return samples["f"][..., None]


def numpyro_spatial_model(
    x: jax.Array,  # [L, D]
    s: jax.Array,  # [L, S]
    f: Optional[jax.Array] = None,
    jitter: float = 1e-5,
):
    """Generic Spatiotemporal model with random spatial effects.

    Args:
        s: Array of input locations, `[S]`.
        x: Array of input covariates, `[S, D]`.
        f: Observed function values, `[S, 1]`.
    """

    L, D = x.shape
    ls = numpyro.sample("ls", dist.Beta(3, 7))
    k = rbf(s, s, var=1.0, ls=ls) + jitter * jnp.eye(L)
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(D), jnp.ones(D)))
    f_mu_x = x @ beta
    f_mu_s = numpyro.sample("f_mu_s", dist.MultivariateNormal(jnp.zeros(L), k))
    f_sigma = numpyro.sample("f_sigma", dist.HalfNormal(0.1))
    numpyro.sample("f", dist.Normal(f_mu_x + f_mu_s, f_sigma))


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
