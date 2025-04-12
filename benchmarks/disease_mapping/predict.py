import pickle
from datetime import datetime
from functools import partial
from pathlib import Path

import arviz as az
import hydra
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC
from omegaconf import DictConfig, OmegaConf

from benchmarks.disease_mapping.data import get_grid, get_population, get_shape
from benchmarks.disease_mapping.samplers import (
    get_np_sampler,
    sample_gp,
    sample_gp_pointwise,
    sample_prevalence,
)
from benchmarks.disease_mapping.utils import batch, map_fn, unbatch
from benchmarks.disease_mapping.visualize import plot_distribution, plot_predictions
from dl4bi.core.train import load_ckpt
from dl4bi.core.utils import breakpoint_if_nonfinite


def predict(seed, model, s_c, samples, s_t, batch_size):
    """
    Transforms samples $y_c (+params) | (s, np, n)_c$ into $y_t | s_t, (s, np, n)_c$.
    """
    rng = jax.random.key(seed)

    y_c = samples.pop("y")
    params = samples
    del samples

    print("Num samples:", y_c.shape[0])
    print("Num context locations:", s_c.shape[0])
    print("Num target locations:", s_t.shape[0])

    s_c = jnp.repeat(s_c[None], batch_size, axis=0)
    s_t = jnp.repeat(s_t[None], batch_size, axis=0)

    y_c = batch(y_c, batch_size)
    params = {k: batch(v, batch_size) for k, v in params.items()}

    num_iters = y_c.shape[0]

    if model.lower() == "gp":
        print("Using GP for predictions.")
        sample_y_t = sample_gp
        model_name = "gp"
    elif model.lower() in ["gp_pointwise", "pointwise", "pointwise_gp"]:
        print("Using GP for pointwise predictions.")
        sample_y_t = sample_gp_pointwise
        model_name = "gp_pointwise"
    else:
        state, cfg_model = load_ckpt(model)
        print(f"Using {cfg_model.name} for predictions.")
        sample_y_t = get_np_sampler(state)  # already batched
        model_name = cfg_model.name

    rng_y_t, rng_theta_t = jax.random.split(rng)

    sample_y_t = partial(sample_y_t, s_c=s_c, s_t=s_t)  # locations are fixed
    y_t = map_fn(sample_y_t)(
        rng=jax.random.split(rng_y_t, num_iters),
        y_c=y_c,
        **params,
    )
    theta_t = map_fn(sample_prevalence)(
        rng=jax.random.split(rng_theta_t, num_iters),
        y=y_t,
        **params,
    )
    theta_t = jax.nn.sigmoid(theta_t)

    breakpoint_if_nonfinite(theta_t)

    return model_name, s_t[0], unbatch(y_t), unbatch(theta_t)


def aggregate(
    samples,  # [..., L],
    populations,  # [L]
):
    return jnp.sum(samples * populations, axis=-1) / jnp.sum(populations)


@hydra.main("configs", "prediction", None)
def main(cfg: DictConfig):
    if cfg.mcmc:
        mcmc_results_path = Path(cfg.mcmc)
    else:
        mcmc_results_path = max(
            Path("results").glob("MCMC*"), key=lambda p: p.stat().st_mtime
        )

    # Load data
    data = dict(jnp.load(mcmc_results_path / "data.npz"))
    with open(mcmc_results_path / "mcmc.pickle", "rb") as f:
        mcmc: MCMC = pickle.load(f)
    samples = mcmc.get_samples()

    # Predict
    s_t = get_grid(cfg.iso, cfg.region, cfg.res)

    t_start = datetime.now()
    model_name, s_t, y_t, theta_t = predict(
        cfg.seed,
        cfg.np,
        data["s"],
        samples,
        s_t,
        cfg.batch_size,
    )
    theta_t = theta_t.block_until_ready()
    t_end = datetime.now()
    print(f"Prediction took {t_end - t_start}.")

    # Save results
    results_path = mcmc_results_path / model_name
    results_path.mkdir(parents=True, exist_ok=True)
    jnp.savez(results_path / "predictions.npz", s=s_t, theta=theta_t)

    # Plot results
    fig = plot_predictions(
        s_t,
        theta_t,
        get_shape(cfg.iso, cfg.region),
        data,
    )
    fig.savefig(
        results_path / "predictions.png",
        dpi=300,
    )

    # Aggregate results
    populations = get_population(cfg.iso, s_t, cfg.res)
    aggregate_samples = aggregate(theta_t, populations)

    jnp.save(results_path / "aggregate_samples.npy", aggregate_samples)
    fig = plot_distribution(aggregate_samples)
    fig.savefig(
        results_path / "aggregate_distribution.png",
        dpi=300,
    )

    # Arviz summary
    num_chains = OmegaConf.load(mcmc_results_path / "config.yaml").num_chains

    def split_chains(samples: jax.Array):
        return samples.reshape(num_chains, -1, *samples.shape[1:])

    posterior = {
        "aggregate_theta_t": aggregate_samples,
        "theta_t": theta_t,
        "y_t": y_t,
    }  # | samples
    # drop mcmc samples here for clarity

    observed_data = {"s_t": s_t} | data

    az_data = az.from_dict(
        posterior=jax.tree.map(split_chains, posterior),
        observed_data=observed_data,
    )
    summary = az.summary(az_data, hdi_prob=0.95)
    print(summary)
    summary.to_csv(results_path / "summary.csv")


if __name__ == "__main__":
    main()
