#!/usr/bin/env python3
import json
from functools import partial
from itertools import product
from pathlib import Path
from time import perf_counter

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
import scoringrules as sr
import wandb
from era5 import dataloader, load_data, plot
from hydra.utils import instantiate
from jax import random
from jax.scipy import linalg as jsp_linalg
from jax.scipy.stats import norm as jax_norm
from numpyro import distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
from omegaconf import DictConfig, OmegaConf
from scipy.stats import norm
from tqdm import tqdm

from dl4bi.core.train import Callback, load_ckpt, save_ckpt, train
from dl4bi.meta_learning.utils import cfg_to_run_name


@hydra.main("configs/era5_bsa_vs_svgp", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", "bsa_tnp_vs_svgp")
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,
    )
    print(OmegaConf.to_yaml(cfg))
    splits = resolve_data_splits(cfg)
    ds_train, ds_valid, ds_test, revert = load_data(**splits)
    svgp_cfg = maybe_sweep_svgp(cfg, ds_valid)
    test_dataloader = partial(dataloader, ds=ds_test, **cfg.benchmark.dataloader)
    records = []
    seed_iter = tqdm(
        cfg.seeds,
        desc="Seeds",
        dynamic_ncols=True,
        disable=not cfg.progress.seeds,
    )
    for seed in seed_iter:
        state, ckpt_path = load_or_train_bsa(seed, cfg, ds_train, ds_valid, revert)
        seed_records = benchmark_seed(seed, state, test_dataloader, cfg, svgp_cfg)
        records.extend(seed_records)
        wandb.log({f"checkpoint/{seed}": str(ckpt_path)})
        for metric, value in summarize_records(seed_records).items():
            wandb.log({f"seed/{seed}/{metric}": value})
    save_results(records, cfg, run_name, svgp_cfg)


def load_or_train_bsa(
    seed: int,
    cfg: DictConfig,
    ds_train,
    ds_valid,
    revert: dict,
):
    bsa_cfg = build_bsa_cfg(cfg, seed)
    run_name = cfg.bsa.get("run_name") or cfg_to_run_name(bsa_cfg)
    ckpt_project = resolve_bsa_project(cfg)
    ckpt_path = Path(cfg.results_dir) / ckpt_project / str(seed) / f"{run_name}.ckpt"
    if ckpt_path.exists():
        state, _ = load_ckpt(ckpt_path, bsa_cfg)
        return state, ckpt_path
    if not cfg.bsa.train_if_missing:
        raise FileNotFoundError(
            f"Missing BSA-TNP checkpoint at {ckpt_path}. "
            "Train it first with benchmarks/meta_learning/era5.py or set "
            "bsa.train_if_missing=True."
        )
    train_dataloader = partial(dataloader, ds=ds_train, **cfg.data.train_dataloader)
    valid_dataloader = partial(dataloader, ds=ds_valid, **cfg.data.valid_dataloader)
    callback_dataloader = partial(
        dataloader,
        ds=ds_valid,
        is_callback=True,
        revert=revert,
        **cfg.data.valid_dataloader,
    )
    optimizer = instantiate(cfg.optimizer)
    model = instantiate(cfg.model)
    rng = random.key(seed)
    state = train(
        rng,
        model,
        optimizer,
        model.train_step,
        cfg.data.train_num_steps,
        train_dataloader,
        model.valid_step,
        cfg.data.valid_interval,
        cfg.data.valid_num_steps,
        valid_dataloader,
        callbacks=[Callback(plot, cfg.data.plot_interval)],
        callback_dataloader=callback_dataloader,
        return_state="best",
    )
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, bsa_cfg, ckpt_path)
    return state, ckpt_path


def build_bsa_cfg(cfg: DictConfig, seed: int):
    return OmegaConf.create(
        {
            "seed": seed,
            "wandb": False,
            "project": resolve_bsa_project(cfg),
            "evaluate_only": False,
            "optimizer": OmegaConf.to_container(cfg.optimizer, resolve=True),
            "model": OmegaConf.to_container(cfg.model, resolve=True),
            "data": {
                **OmegaConf.to_container(cfg.data, resolve=True),
                "splits": resolve_data_splits(cfg),
            },
        }
    )


def resolve_bsa_project(cfg: DictConfig) -> str:
    if cfg.bsa.get("project_name"):
        return cfg.bsa.project_name
    if setting := cfg.get("benchmark_setting"):
        return f"AISTATS BSA-TNP - ERA5 - {setting}"
    return cfg.bsa.get("project_name") or cfg.bsa.get("project") or "ERA5"


def resolve_data_splits(cfg: DictConfig) -> dict:
    default_splits = OmegaConf.to_container(cfg.data.splits, resolve=True)
    setting = cfg.get("benchmark_setting")
    if setting is None:
        return default_splits
    if setting == "CNW":
        return {
            "train_region": "central_europe",
            "valid_region": "northern_europe",
            "test_region": "western_europe",
        }
    if setting == "CWN":
        return {
            "train_region": "central_europe",
            "valid_region": "western_europe",
            "test_region": "northern_europe",
        }
    raise ValueError(
        f"Unsupported benchmark_setting={setting!r}. Expected one of: CNW, CWN."
    )


def benchmark_seed(
    seed: int,
    state,
    test_dataloader,
    cfg: DictConfig,
    svgp_cfg: DictConfig,
):
    rng = random.key(seed)
    batches = test_dataloader(rng)
    records = []
    batch_iter = tqdm(
        range(cfg.benchmark.num_eval_batches),
        total=cfg.benchmark.num_eval_batches,
        desc=f"Seed {seed}",
        leave=False,
        dynamic_ncols=True,
        disable=not cfg.progress.batches,
    )
    for batch_idx in batch_iter:
        batch = next(batches)
        rng_bsa, rng_svgp, rng = random.split(rng, 3)
        bsa_output = state.apply_fn(
            {"params": state.params, **state.kwargs},
            **batch,
            training=False,
            rngs={"extra": rng_bsa},
        )
        if isinstance(bsa_output, tuple):
            bsa_output, _ = bsa_output
        task_records = benchmark_batch(
            batch,
            bsa_output.mu,
            bsa_output.std,
            seed,
            batch_idx,
            rng_svgp,
            cfg,
            svgp_cfg,
        )
        records.extend(task_records)
    return records


def benchmark_batch(
    batch,
    bsa_mu: jax.Array,
    bsa_std: jax.Array,
    seed: int,
    batch_idx: int,
    rng: jax.Array,
    cfg: DictConfig,
    svgp_cfg: DictConfig,
):
    batch_size = batch.f_test.shape[0]
    rngs = random.split(rng, batch_size)
    records = []
    for task_idx in range(batch_size):
        task = extract_task(batch, task_idx)
        target = np.asarray(task["y_test"])
        bsa_metrics = compute_metrics(
            target,
            np.asarray(bsa_mu[task_idx][task["mask_test"], 0]),
            np.asarray(bsa_std[task_idx][task["mask_test"], 0]),
            cfg.benchmark.hdi_prob,
        )
        records.append(
            {
                "seed": seed,
                "batch": batch_idx,
                "task": task_idx,
                "method": "BSA-TNP",
                **bsa_metrics,
            }
        )
        svgp_mu, svgp_std, loss, fit_stats = fit_svgp_predict(
            rngs[task_idx], task, svgp_cfg
        )
        svgp_metrics = compute_metrics(
            target,
            np.asarray(svgp_mu),
            np.asarray(svgp_std),
            cfg.benchmark.hdi_prob,
        )
        records.append(
            {
                "seed": seed,
                "batch": batch_idx,
                "task": task_idx,
                "method": "SVGP",
                "ELBO": float(loss),
                **fit_stats,
                **svgp_metrics,
            }
        )
    return records


def extract_task(batch, idx: int):
    mask_ctx = np.asarray(batch.mask_ctx[idx], dtype=bool)
    mask_test = np.asarray(batch.mask_test[idx], dtype=bool)
    x_ctx = None if batch.x_ctx is None else batch.x_ctx[idx][mask_ctx]
    x_test = None if batch.x_test is None else batch.x_test[idx][mask_test]
    return {
        "x_ctx": stack_features(x_ctx, batch.s_ctx[idx][mask_ctx], batch.t_ctx[idx][mask_ctx]),
        "y_ctx": batch.f_ctx[idx][mask_ctx, 0],
        "x_test": stack_features(
            x_test,
            batch.s_test[idx][mask_test],
            batch.t_test[idx][mask_test],
        ),
        "y_test": batch.f_test[idx][mask_test, 0],
        "mask_test": mask_test,
    }


def stack_features(*arrays):
    parts = [jnp.asarray(a) for a in arrays if a is not None]
    return jnp.concatenate(parts, axis=-1)


def fit_svgp_predict(rng: jax.Array, task: dict, cfg: DictConfig):
    x_ctx = jnp.asarray(task["x_ctx"])
    y_ctx = jnp.asarray(task["y_ctx"])
    x_test = jnp.asarray(task["x_test"])
    rng_z, rng_svi = random.split(rng)
    z_init = initialize_inducing_inputs(rng_z, x_ctx, cfg.num_inducing)
    svi = SVI(svgp_model, svgp_guide, Adam(cfg.learning_rate), Trace_ELBO())
    fit_start = perf_counter()
    svi_state = svi.init(
        rng_svi,
        x_ctx,
        y_ctx,
        z_init,
        cfg.init_amplitude,
        cfg.init_noise,
        cfg.init_lengthscale,
        cfg.learn_inducing_locations,
        cfg.jitter,
    )
    loss = float("nan")
    for _ in range(cfg.num_steps):
        svi_state, loss = svi.update(
            svi_state,
            x_ctx,
            y_ctx,
            z_init,
            cfg.init_amplitude,
            cfg.init_noise,
            cfg.init_lengthscale,
            cfg.learn_inducing_locations,
            cfg.jitter,
        )
    fit_time_s = perf_counter() - fit_start
    params = svi.get_params(svi_state)
    predict_start = perf_counter()
    mu, std = svgp_predictive(x_test, params, z_init, cfg.jitter)
    predict_time_s = perf_counter() - predict_start
    return (
        mu,
        std,
        loss,
        {
            "fit_time_s": fit_time_s,
            "predict_time_s": predict_time_s,
            "total_time_s": fit_time_s + predict_time_s,
            "num_ctx": int(x_ctx.shape[0]),
            "num_test": int(x_test.shape[0]),
            "num_inducing": int(z_init.shape[0]),
            "num_steps": int(cfg.num_steps),
        },
    )


def initialize_inducing_inputs(rng: jax.Array, x: jax.Array, num_inducing: int):
    n = x.shape[0]
    m = min(num_inducing, n)
    idx = random.choice(rng, n, (m,), replace=False)
    return x[idx]


def svgp_model(
    x: jax.Array,
    y: jax.Array,
    z_init: jax.Array,
    init_amplitude: float,
    init_noise: float,
    init_lengthscale: float,
    learn_inducing_locations: bool,
    jitter: float,
):
    dim = x.shape[-1]
    amplitude = numpyro.param(
        "amplitude",
        jnp.array(init_amplitude),
        constraint=constraints.positive,
    )
    noise = numpyro.param(
        "noise",
        jnp.array(init_noise),
        constraint=constraints.positive,
    )
    lengthscale = numpyro.param(
        "lengthscale",
        jnp.full((dim,), init_lengthscale),
        constraint=constraints.positive,
    )
    z = (
        numpyro.param("inducing_inputs", z_init)
        if learn_inducing_locations
        else z_init
    )
    k_zz = rbf_kernel(z, z, amplitude, lengthscale)
    k_zz = k_zz + jitter * jnp.eye(z.shape[0], dtype=x.dtype)
    chol = jnp.linalg.cholesky(k_zz)
    u = numpyro.sample(
        "u",
        dist.MultivariateNormal(
            loc=jnp.zeros(z.shape[0], dtype=x.dtype),
            scale_tril=chol,
        ),
    )
    alpha = jsp_linalg.cho_solve((chol, True), u)
    k_xz = rbf_kernel(x, z, amplitude, lengthscale)
    proj = jsp_linalg.solve_triangular(chol, k_xz.T, lower=True)
    mean = k_xz @ alpha
    var = amplitude - jnp.sum(proj**2, axis=0) + noise**2
    numpyro.sample(
        "obs",
        dist.Normal(mean, jnp.sqrt(jnp.clip(var, a_min=jitter))),
        obs=y,
    )


def svgp_guide(
    x: jax.Array,
    y: jax.Array,
    z_init: jax.Array,
    init_amplitude: float,
    init_noise: float,
    init_lengthscale: float,
    learn_inducing_locations: bool,
    jitter: float,
):
    del x, y, init_amplitude, init_noise, init_lengthscale, learn_inducing_locations, jitter
    m = z_init.shape[0]
    loc = numpyro.param("u_loc", jnp.zeros(m))
    scale_tril = numpyro.param(
        "u_scale_tril",
        1.0e-2 * jnp.eye(m),
        constraint=constraints.lower_cholesky,
    )
    numpyro.sample("u", dist.MultivariateNormal(loc=loc, scale_tril=scale_tril))


def svgp_predictive(
    x_test: jax.Array,
    params: dict,
    z_init: jax.Array,
    jitter: float,
):
    amplitude = params["amplitude"]
    noise = params["noise"]
    lengthscale = params["lengthscale"]
    z = params.get("inducing_inputs", z_init)
    u_loc = params["u_loc"]
    u_scale_tril = params["u_scale_tril"]
    k_zz = rbf_kernel(z, z, amplitude, lengthscale)
    k_zz = k_zz + jitter * jnp.eye(z.shape[0], dtype=x_test.dtype)
    chol = jnp.linalg.cholesky(k_zz)
    k_tz = rbf_kernel(x_test, z, amplitude, lengthscale)
    alpha = jsp_linalg.cho_solve((chol, True), u_loc)
    proj = jsp_linalg.solve_triangular(chol, k_tz.T, lower=True)
    mean = k_tz @ alpha
    prior_diag = amplitude - jnp.sum(proj**2, axis=0)
    precision_proj = jsp_linalg.cho_solve((chol, True), k_tz.T).T
    s = u_scale_tril @ u_scale_tril.T
    var_q = jnp.sum((precision_proj @ s) * precision_proj, axis=1)
    var = prior_diag + var_q + noise**2
    std = jnp.sqrt(jnp.clip(var, a_min=jitter))
    return mean, std


def rbf_kernel(
    x_a: jax.Array,
    x_b: jax.Array,
    amplitude: jax.Array,
    lengthscale: jax.Array,
):
    x_a = x_a / lengthscale
    x_b = x_b / lengthscale
    sq_dist = jnp.sum((x_a[:, None, :] - x_b[None, :, :]) ** 2, axis=-1)
    return amplitude * jnp.exp(-0.5 * sq_dist)


def compute_metrics(
    y: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    hdi_prob: float,
):
    std = np.clip(std, 1.0e-6, None)
    alpha = 1.0 - hdi_prob
    z = abs(norm.ppf(alpha / 2.0))
    lower, upper = mu - z * std, mu + z * std
    return {
        "NLL": float(-jax.device_get(jax_norm.logpdf(y, mu, std)).mean()),
        "RMSE": float(np.sqrt(np.square(y - mu).mean())),
        "MAE": float(np.abs(y - mu).mean()),
        "Coverage": float(((y >= lower) & (y <= upper)).mean()),
        "CRPS": float(np.mean(sr.crps_normal(y, mu, std))),
        "IS": float(np.mean(sr.interval_score(y, lower, upper, alpha))),
    }


def summarize_records(records: list[dict]):
    df = pd.DataFrame(records)
    metric_cols = [
        c
        for c in [
            "NLL",
            "RMSE",
            "MAE",
            "Coverage",
            "CRPS",
            "IS",
            "ELBO",
            "fit_time_s",
            "predict_time_s",
            "total_time_s",
            "num_ctx",
            "num_test",
            "num_inducing",
            "num_steps",
        ]
        if c in df.columns
    ]
    summary = (
        df.groupby("method")[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        f"{row['method']}/{metric}": float(row[metric])
        for row in summary
        for metric in metric_cols
        if pd.notna(row.get(metric))
    }


def save_results(records: list[dict], cfg: DictConfig, run_name: str, svgp_cfg: DictConfig):
    df = pd.DataFrame(records)
    per_seed = (
        df.groupby(["seed", "method"])
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["seed", "method"])
    )
    metrics = [
        c
        for c in [
            "NLL",
            "RMSE",
            "MAE",
            "Coverage",
            "CRPS",
            "IS",
            "ELBO",
            "fit_time_s",
            "predict_time_s",
            "total_time_s",
            "num_ctx",
            "num_test",
            "num_inducing",
            "num_steps",
        ]
        if c in per_seed.columns
    ]
    summary = {}
    grouped = per_seed.groupby("method")
    for method, method_df in grouped:
        summary[method] = {}
        for metric in metrics:
            if method_df[metric].notna().any():
                std = method_df[metric].std(ddof=1)
                std = 0.0 if pd.isna(std) else float(std)
                summary[method][metric] = {
                    "mean": round(float(method_df[metric].mean()), 6),
                    "std": round(std, 6),
                }
    out_dir = Path(cfg.results_dir) / cfg.project / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "task_metrics.csv", index=False)
    per_seed.to_csv(out_dir / "seed_metrics.csv", index=False)
    with open(out_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
    with open(out_dir / "config.json", "w") as fp:
        json.dump(OmegaConf.to_container(cfg, resolve=True), fp, indent=2, sort_keys=True)
    with open(out_dir / "svgp_config.json", "w") as fp:
        json.dump(OmegaConf.to_container(svgp_cfg, resolve=True), fp, indent=2, sort_keys=True)
    wandb.log({"summary_path": str(out_dir / "summary.json")})
    flat_summary = flatten_summary(summary)
    if flat_summary:
        wandb.log(flat_summary)


def flatten_summary(summary: dict):
    flat = {}
    for method, metrics in summary.items():
        for metric, stats in metrics.items():
            for stat, value in stats.items():
                flat[f"summary/{method}/{metric}/{stat}"] = value
    return flat


def maybe_sweep_svgp(cfg: DictConfig, ds_valid):
    sweep_cfg = cfg.get("svgp_sweep")
    if not sweep_cfg or not sweep_cfg.get("enabled", False):
        return OmegaConf.create(OmegaConf.to_container(cfg.svgp, resolve=True))
    candidates = enumerate_svgp_candidates(cfg.svgp, sweep_cfg.grid)
    if len(candidates) == 1:
        return candidates[0]
    valid_dataloader = partial(dataloader, ds=ds_valid, **cfg.benchmark.dataloader)
    best_cfg = None
    best_score = float("inf")
    rows = []
    rng = random.key(sweep_cfg.seed)
    batches = valid_dataloader(rng)
    eval_batches = [next(batches) for _ in range(sweep_cfg.num_batches)]
    for candidate in tqdm(
        candidates,
        desc="SVGP sweep",
        dynamic_ncols=True,
        disable=not cfg.progress.seeds,
    ):
        metrics = evaluate_svgp_candidate(eval_batches, candidate, sweep_cfg.metric, rng)
        row = {**OmegaConf.to_container(candidate, resolve=True), **metrics}
        rows.append(row)
        score = metrics[sweep_cfg.metric]
        if score < best_score:
            best_score = score
            best_cfg = candidate
    sweep_df = pd.DataFrame(rows).sort_values(sweep_cfg.metric)
    wandb.log(
        {
            f"svgp_sweep/best/{k}": v
            for k, v in OmegaConf.to_container(best_cfg, resolve=True).items()
        }
    )
    wandb.log({f"svgp_sweep/best/{sweep_cfg.metric}": best_score})
    if not sweep_df.empty:
        table = wandb.Table(dataframe=sweep_df)
        wandb.log({"svgp_sweep/results": table})
    return best_cfg


def enumerate_svgp_candidates(base_cfg: DictConfig, grid_cfg: DictConfig):
    base = OmegaConf.to_container(base_cfg, resolve=True)
    grid = OmegaConf.to_container(grid_cfg, resolve=True)
    keys = sorted(grid)
    values = [grid[k] for k in keys]
    candidates = []
    for combo in product(*values):
        candidate = dict(base)
        candidate.update(dict(zip(keys, combo, strict=True)))
        candidates.append(OmegaConf.create(candidate))
    return candidates


def evaluate_svgp_candidate(
    eval_batches: list,
    candidate: DictConfig,
    metric: str,
    rng: jax.Array,
):
    records = []
    for batch in eval_batches:
        batch_size = batch.f_test.shape[0]
        rngs = random.split(rng, batch_size + 1)
        rng = rngs[-1]
        for task_idx in range(batch_size):
            task = extract_task(batch, task_idx)
            target = np.asarray(task["y_test"])
            mu, std, loss, fit_stats = fit_svgp_predict(rngs[task_idx], task, candidate)
            records.append(
                {
                    "NLL": compute_metrics(
                        target,
                        np.asarray(mu),
                        np.asarray(std),
                        0.95,
                    )["NLL"],
                    "ELBO": float(loss),
                    **fit_stats,
                }
            )
    df = pd.DataFrame(records)
    return {
        metric: float(df[metric].mean()),
        "ELBO": float(df["ELBO"].mean()),
        "fit_time_s": float(df["fit_time_s"].mean()),
        "total_time_s": float(df["total_time_s"].mean()),
    }


if __name__ == "__main__":
    main()
