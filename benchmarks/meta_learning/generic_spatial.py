#!/usr/bin/env python3
import pickle
import re
from contextlib import redirect_stdout
from functools import partial
from io import StringIO
from pathlib import Path
from typing import Callable, Optional

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import scoringrules as sr
import wandb
from hydra.utils import instantiate
from jax import jit, random, vmap
from jax.experimental import enable_x64
from numpyro.infer import MCMC, NUTS
from omegaconf import DictConfig, OmegaConf
from scipy.stats import norm
from dl4bi_sps.gp import GP
from dl4bi_sps.kernels import rbf
from dl4bi_sps.priors import Prior

from dl4bi.core.train import (
    TrainState,
    evaluate,
    load_ckpt,
    save_ckpt,
    train,
)
from dl4bi.core.utils import to_native
from dl4bi.meta_learning.data.spatial import SpatialData
from dl4bi.meta_learning.utils import cfg_to_run_name


@hydra.main("configs/generic_spatial", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.infer_with_mcmc:
        run_name = "MCMC - Infer"
    if cfg.infer_with_model:
        run_name += " - Infer"
    run_mode = "online" if cfg.wandb else "disabled"
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=run_mode,
        name=run_name,
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_train, rng_test, rng = random.split(rng, 3)
    dataloader, clbk_dataloader = build_dataloaders(cfg.data)
    optimizer = instantiate(cfg.optimizer)
    model = instantiate(cfg.model)
    output_fn = model.output_fn
    model = model.copy(output_fn=lambda x: output_fn(x, min_std=0.05))
    if cfg.infer_with_model or cfg.infer_with_mcmc:
        rng_b, rng_i, rng = random.split(rng, 3)
        batch, extra = next(clbk_dataloader(rng_b))
        idx, sample = batch.sample_for_inference(rng_i, num_samples=1)[0]
        true_params = {k: v for k, v in extra.items() if k in ["beta", "var", "ls"]}
        if cfg.infer_with_model:
            state, _ = load_ckpt(path.with_suffix(".ckpt"))
            metrics = infer_with_model(rng_i, sample, state)
        if cfg.infer_with_mcmc:
            metrics = infer_with_mcmc(rng_i, sample, cfg.mcmc)
        wandb.log({f"Infer {m}": v for m, v in to_native(metrics).items()})
        wandb.log({f"Infer {p}": v for p, v in to_native(true_params).items()})
        with open(path.parent / "MCMC_sample.pkl", "wb") as fp:
            pickle.dump({k: np.array(v) for k, v in sample.items()}, fp)
        return
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
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloaders(data: DictConfig):
    B, D_x, D_s = data.batch_size, data.num_features, len(data.s)
    L = data.num_ctx_max + data.num_test
    s_min = jnp.array([axis["start"] for axis in data.s])
    s_max = jnp.array([axis["stop"] for axis in data.s])
    # NOTE: pure JAX version is faster
    # prior_pred = jit(Predictive(generic_spatial_model, num_samples=B))
    prior_pred = partial(jax_prior_pred, batch_size=B)
    batchify = jit(lambda v: jnp.repeat(v[None, ...], B, axis=0))

    def dataloader(rng: jax.Array, is_callback: bool = False):
        while True:
            rng_s, rng_x, rng_p, rng_b, rng = random.split(rng, 5)
            s = random.uniform(rng_s, (L, D_s), jnp.float32, s_min, s_max)
            x = random.normal(rng_x, (L, D_x))
            samples = prior_pred(rng_p, x, s)
            f = samples["f"][..., None]  # [L, 1]
            b = SpatialData(x=batchify(x), s=batchify(s), f=f).batch(
                rng_b,
                data.num_ctx_min,
                data.num_ctx_max,
                data.num_test,
                test_includes_ctx=False,
            )
            yield (b, samples) if is_callback else b

    return dataloader, partial(dataloader, is_callback=True)


def jax_prior_pred(
    rng: jax.Array,
    x: jax.Array,
    s: jax.Array,
    batch_size: int,
    jitter: float = 1e-5,
):
    """A pure JAX spatial model with fixed and random spatial effects.

    Relative to the numpyro model, this is a faster implementation because it
    amortizes the Cholesky decomposition by fixing lengthscale and variance
    per batch.
    """
    rng_gp, rng_rest = random.split(rng)
    var = Prior("fixed", {"value": 1.0})
    ls = Prior("beta", {"a": 3.0, "b": 7.0})
    # can't jit; hlo lowering fails on cholesky
    f_mu_s, var, ls, *_ = GP(rbf, var, ls, jitter=jitter).simulate(
        rng_gp, s, batch_size
    )
    f_mu_s = f_mu_s[..., 0]  # [B, L]
    f, beta, f_obs_noise = _jax_prior_pred_helper(rng_rest, x, f_mu_s)
    return {
        "var": var,  # [1]
        "ls": ls,  # [1]
        "beta": beta,  # [B, D_x]
        "f_mu_s": f_mu_s,  # [B, L]
        "f_obs_noise": f_obs_noise,
        "f": f,
    }


@jit
def _jax_prior_pred_helper(rng: jax.Array, x: jax.Array, f_mu_s: jax.Array):
    B, (L, D_x) = f_mu_s.shape[0], x.shape
    rng_beta, rng_sigma, rng_noise = random.split(rng, 3)
    beta = random.normal(rng_beta, (D_x,))
    f_mu_x = beta @ x.T  # beta: [D_x], x: [L, D_x], x.T: [D_x, L], f_mu_x: [L]
    f_obs_noise = 0.1 * jnp.abs(random.normal(rng_sigma, (B,)))  # HalfNormal(0.1)
    f = f_mu_x + f_mu_s + f_obs_noise[:, None] * random.normal(rng_noise, (B, L))
    return f, beta, f_obs_noise


def infer_with_model(
    rng: jax.Array,
    sample: dict,
    state: TrainState,
):
    batch = {k: v[None, ...] for k, v in sample.items()}
    output = state.apply_fn(
        {"params": state.params, **state.kwargs},
        **batch,
        training=False,
        rngs={"extra": rng},
    )
    if isinstance(output, tuple):
        output, _ = output  # latent output not used here
    f_mu, f_std = output.mu, output.std  # [B=1, L_test, D_f=1]
    return compute_inference_metrics(
        sample["f_test"][..., 0],  # [L_test]
        f_mu[0, :, 0],  # [L_test]
        f_std[0, :, 0],  # [L_test]
    )


def infer_with_mcmc(
    rng: jax.Array,
    sample: dict,
    mcmc_cfg: DictConfig,
):
    rng_h, rng_p, rng = random.split(rng, 3)
    mcmc_kwargs = {
        "x": sample["x_ctx"],  # [L_ctx, D_x]
        "s": sample["s_ctx"],  # [L_ctx, D_s]
        "f": sample["f_ctx"][..., 0],  # [L_ctx]
    }
    with enable_x64():
        mcmc = MCMC(
            NUTS(generic_spatial_model),
            num_warmup=mcmc_cfg.num_warmup,
            num_samples=mcmc_cfg.num_samples,
            num_chains=mcmc_cfg.num_chains,
        )
        mcmc.run(rng_h, **mcmc_kwargs)
    print_summary(mcmc, r"^\s+mean|^\s+beta|\s+var|^\s+ls")
    post_samples = mcmc.get_samples()
    pp_kwargs = {
        "s_ctx": sample["s_ctx"],  # [L_ctx, D_s]
        "s_test": sample["s_test"],  # [L_test, D_s]
        "x_test": sample["x_test"],  # [L_test, D_x]
    }
    f_mu, f_std = pointwise_post_pred(
        rng_p,
        **pp_kwargs,
        **post_samples,
    )
    return compute_inference_metrics(
        sample["f_test"][..., 0],  # [L_test]
        f_mu,  # [L_test]
        f_std,  # [L_test]
    )


def print_summary(mcmc, pattern):
    buf = StringIO()
    with redirect_stdout(buf):
        mcmc.print_summary()
    text = buf.getvalue()
    filtered = "\n".join(line for line in text.splitlines() if re.match(pattern, line))
    print(filtered)


def generic_spatial_model(
    x: jax.Array,  # [L, D_x]
    s: jax.Array,  # [L, D_s]
    f: Optional[jax.Array] = None,  # [L, 1]
    jitter: float = 1e-5,
):
    """Generic spatial model with fixed and random spatial effects.

    Args:
        x: Array of input covariates, `[L, D]`.
        s: Array of input locations, `[L, S]`.
        f: Observed function values, `[L, 1]`.
    """

    L, D_x = x.shape
    var = numpyro.deterministic("var", 1.0)
    ls = numpyro.sample("ls", dist.Beta(3.0, 7.0))
    k = rbf(s, s, var, ls) + jitter * jnp.eye(L)
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(D_x), jnp.ones(D_x)))
    f_mu_x = x @ beta
    f_mu_s = numpyro.sample("f_mu_s", dist.MultivariateNormal(0, k))
    f_mu = f_mu_x + f_mu_s
    f_obs_noise = numpyro.sample("f_obs_noise", dist.HalfNormal(0.1))
    numpyro.sample("f", dist.Normal(f_mu, f_obs_noise), obs=f)


def compute_inference_metrics(
    f: jax.Array,  # [L_test]
    f_mu: jax.Array,  # [L_test]
    f_std: jax.Array,  # [L_test]
    hdi_prob: float = 0.95,
):
    assert f.shape == f_mu.shape
    assert f.shape == f_std.shape
    alpha = 1 - hdi_prob
    z_score = jnp.abs(norm.ppf(alpha / 2))
    f_lower, f_upper = f_mu - z_score * f_std, f_mu + z_score * f_std
    m = {}
    m["NLL"] = -np.mean(norm.logpdf(f, f_mu, f_std))
    m["IS"] = np.mean(sr.interval_score(f, f_lower, f_upper, alpha))
    m["CRPS"] = np.mean(sr.crps_normal(f, f_mu, f_std))
    m["CVG"] = np.array(((f >= f_lower) & (f <= f_upper))).mean()
    m["MAE"] = np.abs(f - f_mu).mean()
    m["RMSE"] = np.sqrt(np.square(f - f_mu).mean())
    return m


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


if __name__ == "__main__":
    main()
