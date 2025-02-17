#!/usr/bin/env python
from pathlib import Path

import hydra
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import pandas as pd
import wandb
from inference_models.utils import collect_infer_funcs, compute_metrics, run_mcmc
from jax import jit, random
from jax.experimental import enable_x64
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
# Can you use distance bias on covariates too??
# Can we do House Electricity Consumption dataset?? (one million point GP paper)


@hydra.main("configs/hier_bayes", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    p = f"results/{cfg.project}/{cfg.infer_model}/{cfg.data.name}/{cfg.seed}/{run_name}"
    path = Path(p)
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name + (f" - infer {cfg.infer_seed}" if cfg.infer_compare else ""),
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))
    if cfg.infer_compare:
        with enable_x64():
            return compare_inference(path, cfg)
    rng = random.key(cfg.seed)
    dataloader, *_ = collect_infer_funcs(cfg.infer_model, cfg.data)
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


def compare_inference(path: Path, cfg: DictConfig):
    rng = random.key(cfg.infer_seed)
    rng_sample, rng_extra, rng_mcmc, rng_post = random.split(rng, 4)
    cfg.data.batch_size = 1
    cfg.data.num_ctx.min = cfg.infer.num_ctx
    cfg.data.num_ctx.max = cfg.infer.num_ctx
    dataloader, *infer_funcs = collect_infer_funcs(cfg.infer_model, cfg.data)
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
    batch_to_infer_kwargs, numpyro_model, numpyro_pointwise_post_pred, visualize = (
        infer_funcs
    )
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
        "valid_lens_ctx": Nc,
        "s": s,
        "f": f,
        "f_mu_model": f_mu_model,
        "f_std_model": f_std_model,
        "f_mu_pyro": f_mu_pyro,
        "f_std_pyro": f_std_pyro,
        "inv_permute_idx": inv_permute_idx,
        **{k + "_true": v for k, v in prior_samples.items()},
        **{k + "_post": v for k, v in post_samples.items()},
    }
    jnp.save(path.with_suffix(f".{cfg.infer_seed}.npy"), results)
    metrics = compute_metrics(**results)
    data = []
    for metric, models in metrics.items():
        for model, value in models.items():
            data.append([model, metric, value])  # Model is the category (X-axis)
    df = pd.DataFrame(data, columns=["Model", "Metric", "Value"])
    df["Value"] = df["Value"].astype(float)
    df = df.pivot(index="Metric", columns="Model", values="Value")
    df = df.reset_index()
    print(df)
    wandb.log({"metrics": wandb.Table(dataframe=df)})
    fig = visualize(**results)
    fig.tight_layout()
    fig_path = path.with_suffix(f".{cfg.infer_seed}.png")
    fig.savefig(fig_path)
    plt.close(fig)
    wandb.log({"Posterior Predictive Comparison": wandb.Image(str(fig_path))})


if __name__ == "__main__":
    main()
