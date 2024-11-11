import pickle

#!/usr/bin/env python3
from pathlib import Path
from typing import Union

import hydra
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import optax
from graph_model_utils import instantiate
from jax import random
from map_utils import get_raw_map_data, process_map
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from omegaconf import DictConfig, OmegaConf
from orbax.checkpoint import PyTreeCheckpointer
from plot_utils import (
    plot_covariance,
    plot_histograms,
    plot_kl_on_map,
    plot_trace,
    plot_violin,
)
from prior_cvae import PriorCVAE
from sps.kernels import matern_1_2, matern_3_2, matern_5_2, periodic, rbf
from sps.priors import Prior
from vae import gp_dataloaders

import wandb
from dl4bi.meta_regression.train_utils import (
    TrainState,
    cosine_annealing_lr,
)


@hydra.main("configs", config_name="default_infer", version_base=None)
def main(cfg: DictConfig):
    model_name = f"VAE_GP_{cfg.model.cls}_{cfg.data.sampling_policy}"
    if cfg.infer.run_gp_baseline:
        model_name = "Baseline_GP"
    run_name = cfg.get("name", f"Infer_{model_name}")
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,
    )
    rng, noise_rng = random.split(random.key(cfg.seed))
    s, _, _ = process_map(cfg.data)
    results_dir = Path(
        f"results/{cfg.project}/{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}"
    )

    gp_loader, _, conditionals_names = gp_dataloaders(rng, cfg, s)
    f_batch, _, conditionals = next(iter(gp_loader))
    conditionals = dict(zip(conditionals_names, conditionals))
    map_data = get_raw_map_data(cfg.data.name)
    # NOTE: idxs taking all LTAs as samples, it's possible to give partial information also
    idxs = jnp.arange(len(map_data))
    priors = {
        "var": dist.HalfNormal(),
        "ls": dist.HalfNormal(),
        "period": dist.HalfNormal(),
        "sigma": dist.Gamma(1, 2),
    }
    kernel = globals()[cfg.kernel.kwargs.kernel.func]
    if cfg.infer.run_gp_baseline:
        inference_model = build_baseline_inference_model(s, idxs, kernel, priors)
    else:
        state, _ = load_ckpt((results_dir / model_name).with_suffix(".ckpt"))
        vae_model = (state, instantiate(cfg.model))
        inference_model = build_surrogate_inference_model(
            vae_model, idxs, kernel, cfg.model.kwargs.z_dim, priors
        )

    f = f_batch[0].squeeze()
    if cfg.data.obs_noise > 0:
        f += random.multivariate_normal(
            noise_rng, jnp.zeros_like(f), cfg.data.obs_noise * jnp.eye(f.shape[0])
        )
    hmc_res = _hmc(cfg, inference_model, s, f, conditionals, idxs)
    log_inference_run(
        hmc_res,
        s,
        f_batch,
        conditionals,
        kernel,
        priors,
        cfg.data.obs_noise,
        map_data,
        model_name,
        results_dir,
    )


def _hmc(cfg, model, s, f, conditionals, idxs):
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    mcmc = MCMC(nuts, **cfg.infer.mcmc)
    mcmc.run(k1, f[idxs].squeeze())
    mcmc.print_summary()
    samples = mcmc.get_samples()
    post = Predictive(model, samples)(k2)
    post.update(
        {
            "s": s,
            "f": f,
            "idxs": idxs,
            **conditionals,
        }
    )
    return samples, mcmc, post


def log_inference_run(
    hmc_res,
    s,
    f_batch,
    conditionals,
    kernel,
    priors,
    obs_noise,
    map_data,
    model_name,
    results_dir,
):
    samples, mcmc, post = hmc_res
    with open(results_dir / f"{model_name}_hmc_samples.pkl", "wb") as out_file:
        pickle.dump(samples, out_file)
    kl_per_location, kl_total = compute_kl_divergence(f_batch, post)
    post.update(
        {"sigma": obs_noise, "kl_per_location": kl_per_location, "kl_mean": kl_total}
    )
    with open(results_dir / f"{model_name}_hmc_pp.pkl", "wb") as out_file:
        pickle.dump(post, out_file)
    plot_trace(samples, mcmc, conditionals, obs_noise, model_name)
    plot_covariance(samples, conditionals, model_name, kernel, s)
    plot_histograms(samples, conditionals, obs_noise, model_name, priors)
    plot_violin(post, f_batch, model_name)
    plot_kl_on_map(map_data, kl_per_location, model_name)


def build_baseline_inference_model(s, idxs, kernel, priors, jitter=1e-4):
    I_jitter = jitter * jnp.eye(s.shape[0])

    def gp_model(y=None):
        variance = numpyro.sample("var", priors["var"])
        lengthscale = numpyro.sample("ls", priors["ls"])
        if kernel.__name__ == "periodic":
            period = numpyro.sample("period", priors["period"])
            K = kernel(s, s, variance, lengthscale, period) + I_jitter
        else:
            K = kernel(s, s, variance, lengthscale) + I_jitter
        mu = numpyro.sample("mu", dist.MultivariateNormal(0, K))
        sigma = numpyro.sample("sigma", priors["sigma"])
        numpyro.sample("obs", dist.Normal(mu[idxs], sigma), obs=y)

    return gp_model


def build_surrogate_inference_model(model, idxs, kernel, z_dim, priors):
    state, module = model

    def z_to_y_hat(z, conditionals):
        return module.decoder.apply(
            {"params": state.params["decoder"], **state.kwargs},
            jnp.hstack([z, jnp.array(conditionals)]),
        )

    z_dist = dist.MultivariateNormal(jnp.zeros(z_dim), jnp.eye(z_dim))

    def gp_deep_sample_model(y=None):
        variance = numpyro.sample("var", priors["var"])
        lengthscale = numpyro.sample("ls", priors["ls"])
        conditionals = [variance, lengthscale]
        if kernel.__name__ == "periodic":
            conditionals += [numpyro.sample("period", priors["period"])]
        z = numpyro.sample("z", z_dist)
        # NOTE: hyperparams must be added in the same order as conditional names
        # because the underlying VAE assumes they are stacked that way
        mu = numpyro.deterministic("mu", z_to_y_hat(z, conditionals))
        sigma = numpyro.sample("sigma", priors["sigma"])
        numpyro.sample("obs", dist.Normal(mu[idxs], sigma), obs=y)

    return gp_deep_sample_model


def compute_kl_divergence(f_batch, post):
    obs = post["obs"]
    true_mean = jnp.mean(f_batch, axis=0).squeeze()
    true_var = jnp.var(f_batch, axis=0).squeeze()
    post_mean = jnp.mean(obs, axis=0).squeeze()
    post_var = jnp.var(obs, axis=0).squeeze()
    kl_per_location = (
        jnp.log(jnp.sqrt(post_var) / jnp.sqrt(true_var))
        + (true_var + (true_mean - post_mean) ** 2) / (2 * post_var)
        - 0.5
    )
    kl_total = jnp.mean(kl_per_location)
    return kl_per_location, kl_total


def load_ckpt(path: Union[str, Path]):
    "Load a checkpoint."
    if not isinstance(path, Path):
        path = Path(path)
    ckptr = PyTreeCheckpointer()
    ckpt = ckptr.restore(path.absolute())
    cfg = OmegaConf.create(ckpt["config"])
    model = instantiate(cfg.model)
    state = TrainState.create(
        apply_fn=model.apply,
        tx=optax.yogi(cosine_annealing_lr()),
        params=ckpt["state"]["params"],
        kwargs=ckpt["state"]["kwargs"],
    )
    return state, cfg


if __name__ == "__main__":
    main()
