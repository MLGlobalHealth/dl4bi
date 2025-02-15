#!/usr/bin/env python
import importlib
from pathlib import Path
from typing import Callable

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import wandb
from jax import jit, random
from jax.experimental import enable_x64
from jax.scipy.stats import norm
from numpyro.infer import MCMC, NUTS
from omegaconf import DictConfig, OmegaConf

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
# Make 2D plots
# KL(p || q) and KL(q || p)
# Sampling comparisons?
# Can you use distance bias on covariates too??
# Can we do House Electricity Consumption dataset?? (one million point GP paper)


@hydra.main("configs/hier_bayes", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    path = f"results/{cfg.project}/{cfg.inference_model}/{cfg.data.name}/{cfg.seed}/{run_name}"
    path = Path(path)
    wandb_mode = "disabled"
    if cfg.wandb and not cfg.compare_inference:
        wandb_mode = "online"
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=wandb_mode,
        name=run_name,
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    if cfg.compare_inference:
        with enable_x64():
            return compare_inference(rng, path, cfg)
    dataloader, *_ = collect_infer_funcs(cfg.inference_model, cfg.data)
    rng_train, rng_test = random.split(rng)
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


def collect_infer_funcs(model_name: str, data: DictConfig):
    module = importlib.import_module(f"inference_models.{model_name}")
    dataloader = module.build_dataloader(module.jax_prior_pred, data)
    return (
        dataloader,
        module.batch_to_infer_kwargs,
        module.numpyro_model,
        module.numpyro_pointwise_post_pred,
    )


def compare_inference(rng: jax.Array, path: Path, cfg: DictConfig):
    rng_sample, rng_extra, rng_mcmc, rng_post = random.split(rng, 4)
    cfg.data.batch_size = 1
    cfg.data.num_ctx.min = cfg.infer.num_ctx
    cfg.data.num_ctx.max = cfg.infer.num_ctx
    dataloader, *infer_funcs = collect_infer_funcs(cfg.inference_model, cfg.data)
    state, _ = load_ckpt(path.with_suffix(".ckpt"))
    batch = next(dataloader(rng_sample))
    (
        s_ctx,
        f_ctx,
        valid_lens_ctx,
        s,
        f,
        valid_lens_test,
        inv_permute_idx,
        prior_samples,
    ) = batch
    f_mu_model, f_std_model = jit(state.apply_fn)(
        {"params": state.params, **state.kwargs},
        s_ctx,
        f_ctx,
        s,
        valid_lens_ctx,
        valid_lens_test,
        rngs={"extra": rng_extra},
    )
    batch_to_infer_kwargs, numpyro_model, numpyro_pointwise_post_pred = infer_funcs
    rng_sample, rng_mcmc, rng_post, rng = random.split(rng, 4)
    batch = next(dataloader(rng_sample))
    kwargs = batch_to_infer_kwargs(batch, cfg.data, cfg.infer)
    mcmc = run_mcmc(rng_mcmc, numpyro_model, cfg.infer.mcmc, **kwargs)
    mcmc.print_summary()
    post_samples = mcmc.get_samples()
    f_mu_pyro, f_std_pyro = numpyro_pointwise_post_pred(
        rng_post,
        **kwargs,
        **post_samples,
    )
    S, Nc = len(cfg.data.s), valid_lens_ctx[0]
    s_ctx = s_ctx[0, :Nc, :S].squeeze()  # if s is S + X, only take locations
    f_ctx = f_ctx[0, :Nc, 0]
    s = s[0, :, :S].squeeze()
    f = f[0, :, 0]
    f_mu_model = f_mu_model[0, :, 0]
    f_std_model = f_std_model[0, :, 0]
    f_mu_model = f_mu_model.at[:Nc].set(f_ctx)
    f_std_model = f_std_model.at[:Nc].set(0.0)
    f_mu_pyro = jnp.hstack([f_ctx, f_mu_pyro])
    f_std_pyro = jnp.hstack([jnp.zeros(Nc), f_std_pyro])
    results = {
        "s_ctx": s_ctx,
        "f_ctx": f_ctx,
        "valid_lens_ctx": valid_lens_ctx,
        "s": s[inv_permute_idx],
        "f": f[inv_permute_idx],
        "f_mu_model": f_mu_model[inv_permute_idx],
        "f_std_model": f_std_model[inv_permute_idx],
        "f_mu_pyro": f_mu_pyro[inv_permute_idx],
        "f_std_pyro": f_std_pyro[inv_permute_idx],
        "inv_permute_idx": inv_permute_idx,
        **{k + "_true": v for k, v in prior_samples.items()},
        **{k + "_post": v for k, v in post_samples.items()},
    }
    jnp.save(path.with_suffix(".npy"), results)
    plot_posterior_predictive(
        s_ctx,
        f_ctx,
        s[inv_permute_idx],
        f[inv_permute_idx],
        f_mu_model[inv_permute_idx],
        f_std_model[inv_permute_idx],
        f_mu_pyro[inv_permute_idx],
        f_std_pyro[inv_permute_idx],
    )


def run_mcmc(rng: jax.Array, model: Callable, infer: DictConfig, **kwargs):
    mcmc = MCMC(
        NUTS(model),
        num_warmup=infer.num_warmup,
        num_samples=infer.num_samples,
        num_chains=infer.num_chains,
    )
    mcmc.run(rng, **kwargs)
    return mcmc


def plot_posterior_predictive(
    s_ctx: jax.Array,  # [L_ctx, S]
    f_ctx: jax.Array,  # [L_ctx, 1]
    s: jax.Array,  # [L, S]
    f: jax.Array,  # [L, 1]
    f_mu_model: jax.Array,  # [L, 1]
    f_std_model: jax.Array,  # [L, 1]
    f_mu_pyro: jax.Array,  # [L, 1]
    f_std_pyro: jax.Array,  # [L, 1]
    hdi_prob: float = 0.95,
):
    # palette from https://davidmathlogic.com/colorblind
    magenta, green, blue, gold = "#D81B60", "#D81B60", "#1E88E5", "#FFC107"
    if s_ctx.ndim == 1:
        plt.scatter(s_ctx, f_ctx, color=magenta)
        plt.plot(s, f, color=magenta)
        _plot_bounds(s, f_mu_model, f_std_model, color=blue)
        _plot_bounds(s, f_mu_pyro, f_std_pyro, color=gold)
        plt.xlabel("s")
        plt.ylabel("f")
        plt.title("GP 1D")
        plt.savefig("/tmp/test.pdf")
    else:  # 2D
        # TODO(danj): implement 2D!!
        pass


def _plot_bounds(
    s: jax.Array,
    f_mu: jax.Array,
    f_std: jax.Array,
    color: str = "steelblue",
    hdi_prob: float = 0.95,
):
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    f_lower = f_mu - z_score * f_std
    f_upper = f_mu + z_score * f_std
    plt.plot(s, f_mu, color=color)
    plt.fill_between(s, f_lower, f_upper, alpha=0.4, color=color, interpolate=True)


if __name__ == "__main__":
    main()
