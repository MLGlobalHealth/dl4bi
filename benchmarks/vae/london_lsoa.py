import sys

sys.path.append("benchmarks/vae")
import pickle
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

import arviz as az
import flax.linen as nn
import geopandas as gpd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import optax
import pandas as pd
from jax import Array, jit, random, value_and_grad
from matplotlib.lines import Line2D
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value
from omegaconf import DictConfig
from shapely.affinity import scale, translate
from sps.kernels import matern_1_2
from utils.plot_utils import (
    plot_infer_trace,
    plot_models_predictive_means,
    plot_posterior_predictive_comparisons,
)

import wandb
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import TrainState, cosine_annealing_lr, evaluate, save_ckpt, train
from dl4bi.vae import gMLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step, generate_surrogate_decoder


def main(data_type, seed=59, num_chains=2, obs_ratio=0.5):
    wandb.init(mode="disabled")
    rng = random.key(seed)
    rng_train, rng_test, rng_infer, rng_idxs, rng_init = random.split(rng, 5)
    save_dir = Path(f"results/{data_type}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    map_data = gpd.read_file(f"benchmarks/vae/maps/{data_type}")
    s_max = 100
    s = gen_spatial_structure(map_data, s_max=s_max)
    models = {"DeepRV + gMLP": gMLPDeepRV(num_blks=2), "Baseline_GP": None}
    y_obs = jnp.array(map_data["data"], dtype=jnp.float32)
    population = jnp.array(map_data.population, dtype=jnp.int32)
    obs_mask = gen_random_obs_mask(rng_idxs, s, obs_ratio)
    priors, init_vals = joint_map_init(
        rng_init, s, population, y_obs, obs_mask, num_chains
    )
    infer_model, cond_names = inference_model(s, priors, population)
    loader = gen_train_dataloader(
        s,
        {"ls": dist.Uniform(1.0, s_max / 2 + 5.0)},
        batch_size=32 if s.shape[0] < 2500 else 16,
    )
    y_hats, all_samples, result = [y_obs], [], []
    for model_name, model in models.items():
        model_path = save_dir / f"{model_name}"
        # if (model_path / "single_res.pkl").exists():
        #     with open(model_path / "hmc_samples.pkl", "rb") as out_file:
        #         samples = pickle.load(out_file)
        #     with open(model_path / "hmc_pp.pkl", "rb") as out_file:
        #         post = pickle.load(out_file)
        #     with open(model_path / "single_res.pkl", "rb") as out_file:
        #         res = pickle.load(out_file)
        #     y_hats.append(post["obs"])
        #     all_samples.append(samples)
        #     result.append(res)
        #     continue
        model_path.mkdir(parents=True, exist_ok=True)
        train_time, eval_mse, surrogate_decoder = None, None, None
        if model is not None:
            train_time, eval_mse, surrogate_decoder = (
                load_surr_model_results(model, model_name, data_type)
                if (model_path / "model.ckpt").exists()
                else surrogate_model_train(
                    rng_train,
                    rng_test,
                    s.shape[0],
                    loader,
                    model,
                    save_dir / f"{model_name}",
                )
            )
        samples, mcmc, post, infer_time = hmc(
            rng_infer,
            infer_model,
            y_obs,
            obs_mask,
            model_path,
            surrogate_decoder,
            init_vals,
        )
        y_hats.append(post["obs"])
        all_samples.append(samples)
        ess = az.ess(mcmc, method="mean")
        try:
            plot_infer_trace(
                samples, mcmc, None, cond_names, model_path / "infer_trace.png"
            )
        except Exception:
            pass
        res = {
            "model_name": model_name,
            "seed": seed,
            "train_time": train_time,
            "Test Norm MSE": eval_mse,
            "infer_time": infer_time,
            "MSE(y, y_hat)": ((y_obs - post["obs"].mean(axis=0)) ** 2).mean(),
        }
        res.update({f"inferred {c} mean": samples[c].mean(axis=0) for c in cond_names})
        res.update(
            {f"ESS {c}": ess[c].mean().item() if ess else None for c in cond_names}
        )
        with open(model_path / "single_res.pkl", "wb") as out_file:
            pickle.dump(res, out_file)
        result.append(res)
    scatter_plot_model_vs_model(all_samples, list(models.keys()), obs_mask, save_dir)
    plot_models_predictive_means(
        y_hats, map_data, save_dir / "obs_means.png", obs_mask, log=True
    )
    scatter_plot_prevalence(
        y_obs, population, all_samples, list(models.keys()), obs_mask, save_dir
    )
    plot_posterior_predictive_comparisons(
        all_samples,
        {},
        priors,
        list(models.keys()),
        cond_names,
        save_dir / "comp",
        baseline_model="DeepRV + gMLP" if s.shape[0] > 2500 else "Baseline_GP",
    )
    pd.DataFrame(result).to_csv(save_dir / "res.csv")


def hmc(
    rng: Array,
    model: Callable,
    y_obs: Array,
    obs_mask: Union[Array, bool],
    results_dir: Path,
    surrogate_decoder: Optional[Callable] = None,
    init_vals: Optional[list[dict]] = [],
):
    """Run HMC with multiple independent single-chain runs and merge results."""
    num_chains, n_samples, n_warmup = len(init_vals), 4000, 4000
    if y_obs.shape[0] > 2500 and surrogate_decoder is None:
        num_chains, n_samples, n_warmup = 2, 1000, 500
    all_samples, all_posts = [], []
    total_time = 0.0

    for i in range(num_chains):
        chain_init = init_vals[i]
        print(f"Running chain {i} ...")
        rng, subrng = random.split(rng)
        init_strat = init_to_value(values=chain_init)
        nuts = NUTS(model, init_strategy=init_strat, target_accept_prob=0.9)
        mcmc = MCMC(nuts, num_chains=1, num_samples=n_samples, num_warmup=n_warmup)
        k1, k2 = random.split(subrng)
        start = datetime.now()
        mcmc.run(k1, surrogate_decoder=surrogate_decoder, y=y_obs, obs_mask=obs_mask)
        total_time += (datetime.now() - start).total_seconds()
        samples = mcmc.get_samples()
        post = Predictive(model, samples)(k2, surrogate_decoder=surrogate_decoder)
        all_samples.append(samples)
        all_posts.append(post)
    combined_samples = {
        k: jnp.concatenate([s[k] for s in all_samples], axis=0) for k in all_samples[0]
    }
    combined_post = {
        k: jnp.concatenate([p[k] for p in all_posts], axis=0) for k in all_posts[0]
    }
    idata = az.from_dict(
        posterior={
            k: it.reshape((num_chains, n_samples, -1)).squeeze()
            for k, it in combined_samples.items()
        },
        posterior_predictive={
            k: it.reshape((num_chains, n_samples, -1)).squeeze()
            for k, it in combined_post.items()
        },
    )
    combined_post["infer_time"] = total_time
    with open(results_dir / "hmc_samples.pkl", "wb") as out_file:
        pickle.dump(combined_samples, out_file)
    with open(results_dir / "hmc_pp.pkl", "wb") as out_file:
        pickle.dump(combined_post, out_file)
    return combined_samples, idata, combined_post, total_time


def surrogate_model_train(
    rng_train: Array,
    rng_test: Array,
    grid_size: int,
    loader: Callable,
    model: nn.Module,
    results_dir: Path,
    train_num_steps: int = 500_000,
    valid_interval: int = 100_000,
    valid_steps: int = 5_000,
    state=None,
):
    train_step = deep_rv_train_step
    train_num_steps = train_num_steps if state is None else 120_000
    lr = 1e-3 if grid_size < 2500 else 2e-3
    train_num_steps = (
        train_num_steps if grid_size < 2500 else int(train_num_steps * 1.2)
    )
    if state is not None:
        lr = lr / 10
    optimizer = optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.adamw(cosine_annealing_lr(train_num_steps, lr), weight_decay=1e-2),
    )
    start = datetime.now()
    state = train(
        rng_train,
        model,
        optimizer,
        train_step,
        train_num_steps,
        loader,
        valid_step,
        valid_interval,
        valid_steps,
        loader,
        return_state="best",
        valid_monitor_metric="norm MSE",
        state=state,
    )
    train_time = (datetime.now() - start).total_seconds()
    eval_mse = evaluate(rng_test, state, valid_step, loader, valid_steps)["norm MSE"]
    save_ckpt(state, DictConfig({}), results_dir / "model.ckpt")
    with open(results_dir / "train_metrics.pkl", "wb") as out_file:
        pickle.dump({"train_time": train_time, "eval_mse": eval_mse}, out_file)
    surrogate_decoder = generate_surrogate_decoder(state, model)
    return train_time, eval_mse, surrogate_decoder


def gen_train_dataloader(s: Array, priors: dict[str, dist.Distribution], batch_size):
    jitter = 5e-4 * jnp.eye(s.shape[0])
    kernel_jit = jit(lambda s, var, ls: matern_1_2(s, s, var, ls) + jitter)
    f_jit = jit(lambda K, z: jnp.einsum("ij,bj->bi", jnp.linalg.cholesky(K), z))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z = random.split(rng_data, 3)
            var = 1.0
            ls = priors["ls"].sample(rng_ls)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s.shape[0]))
            K = kernel_jit(s, var, ls)
            f = f_jit(K, z)
            yield {"s": s, "f": f, "z": z, "conditionals": jnp.array([ls])}

    return dataloader


def inference_model(s: Array, priors: dict, population: Array):
    """
    Binomial inference model with spatial GP + covariates (area, avg_room_num, interaction).
    """

    surrogate_kwargs = {"s": s}
    jitter = 5e-4 * jnp.eye(s.shape[0])

    def binomial(surrogate_decoder=None, obs_mask=True, y=None):
        var = numpyro.sample("var", priors["var"])
        ls = numpyro.sample("ls", priors["ls"])
        beta = numpyro.sample("beta", priors["beta"])
        z = numpyro.sample("z", dist.Normal(), sample_shape=(s.shape[0],))
        if surrogate_decoder is None:
            K = matern_1_2(s, s, 1.0, ls) + jitter
            L_chol = jnp.linalg.cholesky(K)
            mu = numpyro.deterministic("mu", jnp.matmul(L_chol, z))
        else:
            mu = numpyro.deterministic(
                "mu",
                surrogate_decoder(
                    z[None], jnp.array([ls]), **surrogate_kwargs
                ).squeeze(),
            )
        eta = jnp.sqrt(var) * mu + beta
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample(
                "obs", dist.Binomial(logits=eta, total_count=population), obs=y
            )

    return binomial, ["var", "ls", "beta"]


@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}


def gen_spatial_structure(map_data: gpd.GeoDataFrame, s_max: int):
    """generates a 0-s_max range locations from the geo-locations centroids"""
    centroids = map_data.geometry.centroid
    minx, maxx = centroids.x.min(), centroids.x.max()
    miny, maxy = centroids.y.min(), centroids.y.max()
    x_tran, x_div = minx, (maxx - minx) / s_max
    y_tran, y_div = miny, (maxy - miny) / s_max

    def norm_geom(geom):
        centered_geom = translate(geom, xoff=-x_tran, yoff=-y_tran)
        normalized_geom = scale(
            centered_geom, xfact=1 / x_div, yfact=1 / y_div, origin=(0, 0)
        )
        return normalized_geom

    norm_map = map_data.copy()
    norm_map["geometry"] = norm_map.geometry.apply(norm_geom)
    centroids = norm_map.geometry.centroid
    return jnp.stack([centroids.x.values, centroids.y.values], axis=-1)


def gen_random_obs_mask(rng: Array, s: Array, obs_ratio: float):
    L = s.shape[0]
    num_obs_locations = int(obs_ratio * L)
    obs_idxs = random.choice(rng, jnp.arange(L), (num_obs_locations,), replace=False)
    return jnp.isin(jnp.arange(L), obs_idxs)


def load_surr_model_results(model, model_name, data_type):
    from orbax.checkpoint import PyTreeCheckpointer

    model_path = Path(f"results/{data_type}/{model_name}").absolute()
    optimizer = optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.adamw(cosine_annealing_lr(500_000, 1e-3), weight_decay=1e-2),
    )
    ckptr = PyTreeCheckpointer()
    ckpt = ckptr.restore(model_path / "model.ckpt")
    state = TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=ckpt["state"]["params"],
        kwargs=ckpt["state"]["kwargs"],
    )
    surrogate_decoder = generate_surrogate_decoder(state, model)
    with open(model_path / "train_metrics.pkl", "rb") as ff:
        train_metrics = pickle.load(ff)
    return train_metrics["train_time"], train_metrics["eval_mse"], surrogate_decoder


def joint_map_init(
    rng,
    s,
    population,
    y_obs,
    obs_mask,
    num_chains,
    ls_init=5.5,
    var_init=0.5,
    beta_init=-1.5,
    opt_steps=1200,
    lr=5e-3,
):
    """
    Joint MAP for z, log_ls, log_var, beta.
    Returns:
      - priors dict (LogNormal for ls/var centered at MAP)
      - init_vals dict, with arrays of shape [num_chains, ...]
    """
    L = s.shape[0]
    N = population
    # param initializations (optimize in unconstrained space)
    z = random.normal(rng, (L,)) * 0.1
    log_ls = jnp.log(ls_init)
    log_var = jnp.log(var_init)
    beta = jnp.array(beta_init)

    def neg_logpost(params):
        z, log_ls, log_var, beta = params
        ls = jnp.exp(log_ls)
        var = jnp.exp(log_var)
        K = matern_1_2(s, s, 1.0, ls) + 5e-4 * jnp.eye(L)
        L_chol = jnp.linalg.cholesky(K)
        mu = L_chol @ z
        # priors
        log_prior_z = -0.5 * jnp.sum(z**2)
        log_prior_logls = -0.5 * ((log_ls - jnp.log(ls_init)) ** 2) / (0.8**2)
        log_prior_logvar = -0.5 * ((log_var - jnp.log(var_init)) ** 2) / (1.0**2)
        log_prior_beta = -0.5 * ((beta - beta_init) ** 2) / (2.0**2)
        # likelihood
        logits = jnp.sqrt(var) * mu + beta
        log_sig = -nn.softplus(-logits)
        log_one_minus_sig = -nn.softplus(logits)
        log_like = jnp.sum(
            obs_mask * (y_obs * log_sig + (N - y_obs) * log_one_minus_sig)
        )
        return -(
            log_prior_z + log_prior_logls + log_prior_logvar + log_prior_beta + log_like
        )

    params = (z, log_ls, log_var, beta)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jit
    def step(params, opt_state):
        loss, grads = value_and_grad(neg_logpost)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # optimize
    for i in range(int(opt_steps * (1 if L < 2500 else 1.5))):
        params, opt_state, loss = step(params, opt_state)
        if (i % 200) == 0:
            print(f"iter {i}, neg_logpost {float(loss)}")
    z_map, log_ls_map, log_var_map, beta_map = params
    ls_map = float(jnp.exp(log_ls_map))
    var_map = float(jnp.exp(log_var_map))
    beta_map = float(beta_map)
    # diagnostics
    L_chol_map = jnp.linalg.cholesky(matern_1_2(s, s, 1.0, ls_map) + 5e-4 * jnp.eye(L))
    p_pred = nn.sigmoid(jnp.sqrt(var_map) * (L_chol_map @ z_map) + beta_map)
    prev_obs = y_obs / N
    unweighted_mse = float(jnp.mean(((p_pred - prev_obs) ** 2)[obs_mask]))
    weighted_mse = float(jnp.sum(N * (p_pred - prev_obs) ** 2) / jnp.sum(N))
    print(f"MAP ls={ls_map:.3f}, var={var_map:.3f}, beta={beta_map:.3f}")
    print(f"MAP unweighted MSE {unweighted_mse:.6f}, weighted MSE {weighted_mse:.6f}")
    print("p_pred min/max:", float(p_pred.min()), float(p_pred.max()))
    print("||z||:", float(jnp.linalg.norm(z_map)))
    priors = {
        "var": dist.LogNormal(jnp.log(max(var_map, 1e-3)), 0.75),
        "ls": dist.Gamma(4, 4 / ls_map),
        "beta": dist.Normal(beta_map, 1.0),
    }
    map_vals = {"var": var_map, "ls": ls_map, "beta": beta_map, "z": z_map}
    min_vals = {"var": 0.1, "ls": 1.2}
    init_vals = []
    for i in range(num_chains):
        init_vals.append({})
        for k in map_vals.keys():
            rng, _ = random.split(rng)
            eps = 1e-2 if k != "z" else 1e-3
            shape = tuple() if k != "z" else map_vals[k].shape
            init_vals[i][k] = map_vals[k] + eps * random.normal(rng, shape)
            if k in min_vals:
                init_vals[i][k] = max(min_vals[k], init_vals[i][k])
        print(
            f"Chain {i} ls={init_vals[i]['ls']:.3f},"
            f"var={init_vals[i]['var']:.3f},"
            f" beta={init_vals[i]['beta']:.3f}"
        )
    return priors, init_vals


def scatter_plot_prevalence(
    y_obs,
    population,
    all_samples,
    model_names: list,
    obs_mask,
    save_dir,
    max_points=100,
    seed=78,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    p_obs = y_obs / population
    p_hat_means = []
    for i in range(len(model_names)):
        mu = jnp.asarray(all_samples[i]["mu"])  # (S, L)
        beta = jnp.asarray(all_samples[i]["beta"])[:, None]  # (S,1)
        var = jnp.asarray(all_samples[i]["var"])[:, None]  # (S,1)
        mean_logit = jnp.mean(jnp.sqrt(var) * mu + beta, axis=0)
        p_hat = nn.sigmoid(mean_logit)
        p_hat_means.append(jnp.clip(p_hat, 0.0, 1.0))
    plot_types = {
        "all": jnp.ones_like(p_obs, dtype=bool),
        "observed": obs_mask,
        "unobserved": ~obs_mask,
    }
    styles = {
        "Baseline_GP": {"edgecolor": "blue", "facecolor": "none", "marker": "o"},
        "DeepRV + gMLP": {"edgecolor": "red", "facecolor": "red", "marker": "x"},
    }
    rng = random.key(seed)
    for name, mask in plot_types.items():
        rng, _ = random.split(rng)
        idxs = random.choice(
            rng, jnp.arange(int(sum(mask))), shape=(max_points,), replace=False
        )
        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
        for i, model_name in enumerate(model_names):
            label = model_name.replace("Baseline_", "")
            style = styles.get(
                model_name, {"edgecolor": "gray", "facecolor": "gray", "marker": "s"}
            )
            if style["marker"] == "x":
                ax.scatter(
                    p_obs[mask][idxs],
                    p_hat_means[i][mask][idxs],
                    label=label,
                    alpha=0.8,
                    s=50,
                    color=style["edgecolor"],
                    marker=style["marker"],
                )
            else:
                ax.scatter(
                    p_obs[mask][idxs],
                    p_hat_means[i][mask][idxs],
                    label=label,
                    alpha=0.8,
                    s=50,
                    facecolors=style["facecolor"],
                    edgecolors=style["edgecolor"],
                    marker=style["marker"],
                    linewidths=1.2,
                )
        min_val = float(jnp.nanmin(p_obs[mask]))
        max_val = float(jnp.nanmax(p_obs[mask]))
        ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1)
        if name == "unobserved" and "DeepRV + gMLP" in model_names:
            x = p_obs[mask]
            y = p_hat_means[model_names.index("DeepRV + gMLP")][mask]
            slope, intercept = jnp.polyfit(x, y, 1)
            ax.plot(
                [min_val, max_val],
                intercept + slope * jnp.array([min_val, max_val]),
                "k-",
                lw=1,
            )
        ax.set_xlabel(r"Observed prevalence ($\frac{y}{N}$)")
        ax.set_ylabel("Predicted prevalence")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(alpha=0.3)
        if len(model_names) > 1:
            ax.legend(loc="upper left", frameon=False)
        fig.savefig(
            save_dir / f"scatter_prevalence_{name}.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)


def scatter_plot_model_vs_model(
    all_samples,
    model_names: list,
    obs_mask,
    save_dir,
    cred: float = 0.5,
    max_points: int = 100,
    seed: int = 0,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    name_x = model_names[0]
    name_y = model_names[1]

    def summarize_posterior(samples_dict):
        mu = jnp.asarray(samples_dict["mu"])
        beta = jnp.asarray(samples_dict["beta"])
        var = jnp.asarray(samples_dict["var"])
        logits = jnp.sqrt(var)[:, None] * mu + beta[:, None]
        p_samples = nn.sigmoid(logits)
        p_mean = jnp.nanmean(p_samples, axis=0)
        alpha = (1.0 - cred) / 2.0
        p_lo = jnp.nanquantile(p_samples, alpha, axis=0)
        p_hi = jnp.nanquantile(p_samples, 1.0 - alpha, axis=0)
        p_mean = jnp.clip(p_mean, 0.0, 1.0)
        p_lo = jnp.clip(p_lo, 0.0, 1.0)
        p_hi = jnp.clip(p_hi, 0.0, 1.0)
        return p_mean, p_lo, p_hi

    x_mean, x_lo, x_hi = summarize_posterior(all_samples[0])
    y_mean, y_lo, y_hi = summarize_posterior(all_samples[1])
    plot_types = {
        "all": jnp.ones_like(x_mean, dtype=bool),
        "observed": obs_mask,
        "unobserved": ~obs_mask,
    }
    rng = np.random.default_rng(seed)
    cred_pct = int(round(cred * 100))
    for name, mask in plot_types.items():
        idx = jnp.where(mask)[0]
        if idx.size > max_points:
            pick = rng.choice(np.array(idx), size=max_points, replace=False)
            pick = jnp.array(pick)
        else:
            pick = idx
        xm = x_mean[pick]
        ym = y_mean[pick]
        xlo = x_lo[pick]
        xhi = x_hi[pick]
        ylo = y_lo[pick]
        yhi = y_hi[pick]
        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
        ax.scatter(
            xm,
            ym,
            s=28,
            alpha=0.85,
            linewidths=0.8,
            edgecolors="black",
            marker="o",
            label=f"{name_x} vs {name_y}",
        )
        min_val = float(jnp.nanmin(jnp.concatenate([xm, ym])))
        max_val = float(jnp.nanmax(jnp.concatenate([xm, ym])))
        pad = 0.02 * max(1e-8, (max_val - min_val))
        lo = max(0.0, min_val - pad)
        hi = min(1.0, max_val + pad)
        raw_xerr = jnp.vstack([xm - xlo, xhi - xm])
        raw_yerr = jnp.vstack([ym - ylo, yhi - ym])
        ax.errorbar(
            xm,
            ym,
            xerr=np.array(raw_xerr),
            fmt="none",
            ecolor="orange",
            elinewidth=0.8,
            capsize=0,
            alpha=0.6,
            zorder=0,
        )
        ax.errorbar(
            xm,
            ym,
            yerr=np.array(raw_yerr),
            fmt="none",
            ecolor="green",
            elinewidth=0.8,
            capsize=0,
            alpha=0.6,
            zorder=0,
        )
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(r"Predicted prevalence ($\mathbf{p}$) DeepRV")
        ax.set_ylabel(r"Predicted prevalence ($\mathbf{p}$) GP")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.3)
        handles = [
            Line2D([0], [0], color="green", lw=1.2, label="GP, 50% BCI"),
            Line2D([0], [0], color="orange", lw=1.2, label="DeepRV, 50% BCI"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False)
        fig.savefig(
            save_dir / f"scatter_model_vs_model_{name}_cred{cred_pct}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    dt = "London_MSOA_education_deprivation_parsed_thrs_0"
    main(seed=81, data_type=dt, num_chains=4, obs_ratio=0.5)
    dt = "London_LSOA_education_deprivation_parsed_thrs_0"
    main(seed=62, data_type=dt, num_chains=4, obs_ratio=0.5)
