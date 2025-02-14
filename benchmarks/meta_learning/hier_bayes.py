#!/usr/bin/env python
import importlib
import pickle
from pathlib import Path
from time import time
from typing import Callable

import hydra
import jax
import jax.numpy as jnp
import optax
import wandb
from jax import jit, random
from numpyro.infer import MCMC, NUTS, Predictive
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
# Do inference comparison
# Can you use distance bias on covariates too??
# Can we do House Electricity Consumption dataset?? (one million point GP paper)
# Customize so inference models can use any kernel? similar to GP code


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
        return compare_inference(
            rng,
            cfg.inference_model,
            model_path,
            cfg.data,
            cfg.infer,
        )
    _, dataloader = import_inference_functions(cfg.inference_model, cfg.data)
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


def import_inference_functions(model_name: str, data: DictConfig):
    module = importlib.import_module(f"inference_models.{model_name}")
    dataloader = module.build_dataloader(module.jax_prior_pred, data)
    return module.numpyro_model, dataloader


def compare_inference(
    rng: jax.Array,
    inference_model_name: str,
    model_path: Path,
    data: DictConfig,
    infer: DictConfig,
):
    Nc, S = infer.num_ctx, len(data.s)
    rng_sample, rng_extra, rng_mcmc, rng_pp = random.split(rng, 4)
    data.batch_size = 1  # only generate one sample
    numpyro_model, dataloader = import_inference_functions(inference_model_name, data)
    batches = dataloader(rng_sample)
    _, _, _, s, f, _, ls, beta, f_obs_noise = next(batches)
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
    mcmc = run_mcmc(rng_mcmc, numpyro_model, _x, _s, _f, Nc, infer.mcmc)
    print(f"Seconds elapsed for MCMC inference: {time() - start}")
    mcmc.print_summary()
    post = mcmc.get_samples()
    post_pred = Predictive(numpyro_model, post)
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
            "f_obs_noise_true": f_obs_noise,
        }
    )
    with open("compare_inference.pkl", "wb") as f:
        pickle.dump(post, f)
    # TODO(danj): follow this tutorial to get predictive: https://num.pyro.ai/en/stable/examples/gp.html
    # TODO(danj): create a simple GP only model
    # TODO(danj): record f_mu for the real model
    # TODO(danj): calculate comparison metrics, LL under real data?


def run_mcmc(rng: jax.Array, model: Callable, mcmc: DictConfig, *args):
    return MCMC(
        NUTS(model),
        num_warmup=mcmc.num_warmup,
        num_samples=mcmc.num_samples,
        num_chains=mcmc.num_chains,
    ).run(rng, *args)


if __name__ == "__main__":
    main()
