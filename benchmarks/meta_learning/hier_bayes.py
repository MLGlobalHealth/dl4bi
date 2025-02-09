#!/usr/bin/env python
import pickle
from functools import partial
from pathlib import Path
from time import time
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
    load_ckpt,
    save_ckpt,
    select_steps,
    train,
)

# TODO:
# Do inference comparison
# Can you use distance bias on covariates too??
# Can we do House Electricity Consumption dataset?? (one million point GP paper)


@hydra.main("configs/hier_bayes", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    path = f"results/{cfg.project}/{cfg.seed}/{run_name}"
    path = Path(path)
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    if cfg.compare_inference:
        model_path = path.with_suffix(".ckpt")
        return compare_inference(rng, model_path, cfg.data, cfg.infer)
    rng_train, rng_test = random.split(rng)
    # NOTE: sampling from pure jax model since it only inverts the GP
    # covariance matrix once per batch; this also means that each batch
    # shares the same GP hyperparams, even though the samples are different;
    # this is different from the numpyro model which samples new hyperparams
    # and inverts a different covariance matrix for every element in the batch
    dataloader = build_dataloader(jax_spatial_prior_pred_f, cfg.data)
    # dataloader = build_dataloader(numpyro_spatial_prior_pred_f, cfg.data)
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
        f, *rest = prior_pred(rng, x, s, B)  # x: [L, D], s: [L, S], f: [B, L, 1]
        x, s = batchify(x), batchify(s)  # x: [B, L, D], s: [B, L, S]
        valid_lens_ctx = random.randint(rng_v, (B,), Nc_min, Nc_max)
        s = jnp.concatenate([s, x], axis=-1)  # [B, L, S + D]
        s_ctx = s[:, :Nc_max, :]
        f_ctx = f[:, :Nc_max, :]
        return s_ctx, f_ctx, valid_lens_ctx, s, f, valid_lens_test, *rest

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
    f_mu_s, _, ls, *_ = GP(rbf).simulate(rng_gp, s, batch_size)
    f, beta, f_sigma = _jax_spatial_prior_pred_rest(rng_rest, x, f_mu_s.squeeze())
    return f[..., None], ls, beta, f_sigma  # f: [B, L, 1]


@jit
def _jax_spatial_prior_pred_rest(rng: jax.Array, x: jax.Array, f_mu_s: jax.Array):
    B, (L, D) = f_mu_s.shape[0], x.shape
    rng_beta, rng_sigma, rng_noise = random.split(rng, 3)
    beta = random.normal(rng_beta, (B, D))
    f_mu_x = beta @ x.T  # [B, L]
    f_sigma = 0.1 * jnp.abs(random.normal(rng_sigma, (B,)))
    f = f_mu_x + f_mu_s + f_sigma[:, None] * random.normal(rng_noise, (B, L))
    return f, beta, f_sigma


@partial(jit, static_argnames=("batch_size",))
def numpyro_spatial_prior_pred_f(
    rng: jax.Array,
    x: jax.Array,  # [L, D]
    s: jax.Array,  # [L, S]
    batch_size: int = 64,
):
    B = batch_size
    prior_pred = Predictive(numpyro_spatial_model, num_samples=B)
    samples = prior_pred(rng, x=x, s=s)
    return samples["f"][..., None], samples["ls"], samples["beta"], samples["f_sigma"]


def numpyro_spatial_model(
    x: jax.Array,  # [L, D]
    s: jax.Array,  # [L, S]
    f: Optional[jax.Array] = None,  # [L, 1]
    num_f_obs: Optional[int] = None,
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
    f_mu = f_mu_x + f_mu_s
    f_sigma = numpyro.sample("f_sigma", dist.HalfNormal(0.1))
    numpyro.sample("f", dist.Normal(f_mu, f_sigma), obs=f)
    # mask = jnp.arange(L) < (num_f_obs or L)
    # with numpyro.handlers.mask(mask=mask):
    #     numpyro.sample("f", dist.Normal(f_mu, f_sigma), obs=f)


def compare_inference(
    rng: jax.Array,
    model_path: Path,
    data: DictConfig,
    infer: DictConfig,
):
    Nc, S = infer.num_ctx, len(data.s)
    rng_sample, rng_extra, rng_mcmc, rng_pp = random.split(rng, 4)
    data.batch_size = 1  # only generate one sample
    dataloader = build_dataloader(numpyro_spatial_prior_pred_f, data)
    batches = dataloader(rng_sample)
    _, _, _, s, f, _, ls, beta, f_sigma = next(batches)
    valid_lens_ctx = jnp.array([Nc])
    s_ctx, f_ctx = s[:, :Nc], f[:, :Nc]
    state, _ = load_ckpt(model_path)
    # compile in first run, time in second
    for i in range(2):
        start = time()
        f_mu_model, f_std_model = jit(state.apply_fn)(
            {"params": state.params, **state.kwargs},
            s_ctx,
            f_ctx,
            s,
            valid_lens_ctx,
            valid_lens_test=None,
            rngs={"extra": rng_extra},
        )
    print(f"Seconds elapsed for model inference: {time() - start}")
    # _s, _x, _f = s[0, :, :S].squeeze(), s[0, :, S:], f[0]
    _s, _x, _f = s[0, :Nc, :S].squeeze(), s[0, :Nc, S:], f[0, :Nc]
    start = time()
    mcmc = run_mcmc(rng_mcmc, numpyro_spatial_model, _x, _s, _f, Nc, infer.mcmc)
    print(f"Seconds elapsed for MCMC inference: {time() - start}")
    mcmc.print_summary()
    post = mcmc.get_samples()
    post_pred = Predictive(numpyro_spatial_model, post)
    post_pred_samples = jit(post_pred)(rng_pp, _x, _s)
    f_pp = post_pred_samples["f"]
    post.update(
        {
            "x": _x,
            "s": _s,
            "Nc": Nc,
            "f_mu_model": f_mu_model.squeeze(),
            "f_std_model": f_std_model.squeeze(),
            "f_mu_mcmc": jnp.mean(f_pp, axis=0),
            "f_std_mcmc": jnp.std(f_pp, axis=0),
            "f_pp": f_pp,
            "f_true": _f,
            "ls_true": ls,
            "beta_true": beta,
            "f_sigma_true": f_sigma,
        }
    )
    with open("compare_inference.pkl", "wb") as f:
        pickle.dump(post, f)
    # TODO(danj): follow this tutorial to get predictive: https://num.pyro.ai/en/stable/examples/gp.html
    # TODO(danj): create a simple GP only model
    # TODO(danj): record f_mu for the real model
    # TODO(danj): calculate comparison metrics, LL under real data?


def run_mcmc(
    rng: jax.Array,
    model: Callable,
    x: jax.Array,
    s: jax.Array,
    f: jax.Array,
    num_f_obs: int,
    cfg: DictConfig,
):
    mcmc = MCMC(
        NUTS(model),
        num_warmup=cfg.num_warmup,
        num_samples=cfg.num_samples,
        num_chains=cfg.num_chains,
    )
    mcmc.run(rng, x, s, f, num_f_obs)
    return mcmc


if __name__ == "__main__":
    main()
