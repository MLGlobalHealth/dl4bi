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
from numpyro.examples.datasets import LYNXHARE, load_dataset
from numpyro.infer import MCMC, NUTS, Predictive
from omegaconf import DictConfig, OmegaConf
from scipy.stats import norm
from sps.gp import GP
from sps.kernels import rbf
from sps.priors import Prior

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


@hydra.main("configs/predator_prey", config_name="default", version_base=None)
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
    prior_pred = jit(Predictive(predator_prey_model, num_samples=B))
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


# TODO(danj): update
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
            NUTS(predator_prey_model),
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


# source: https://num.pyro.ai/en/stable/examples/ode.html
def predator_prey_model(N, y=None):
    # initial population
    z_init = numpyro.sample("z_init", dist.LogNormal(jnp.log(10), 1).expand([2]))
    # measurement times
    ts = jnp.arange(float(N))
    # params alpha, beta, gamma, delta, of dz_dt
    theta = numpyro.sample(
        "theta",
        dist.TruncatedNormal(
            low=0.0,
            loc=jnp.array([1.0, 0.05, 1.0, 0.05]),
            scale=jnp.array([0.5, 0.05, 0.5, 0.05]),
        ),
    )
    # integrate dz/dt, the result will be Nx2
    z = odeint(dz_dt, z_init, ts, theta, rtol=1e-6, atol=1e-5, mxstep=1000)
    # measurement errors
    sigma = numpyro.sample("sigma", dist.LogNormal(-1, 1).expand([2]))
    # measured populations
    numpyro.sample("y", dist.LogNormal(jnp.log(z), sigma), obs=y)


def dz_dt(z, t, theta):
    """
    Lokta-Volterra equations. Real positive parameters `alpha`, `beta`, `gamma`,
    `delta` describes the interactions of two species.
    """
    u = z[0]
    v = z[1]
    alpha, beta, gamma, delta = (
        theta[..., 0],
        theta[..., 1],
        theta[..., 2],
        theta[..., 3],
    )
    du_dt = (alpha - beta * v) * u
    dv_dt = (-gamma + delta * u) * v
    return jnp.stack([du_dt, dv_dt])


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


# TODO(danj): update
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


if __name__ == "__main__":
    main()
