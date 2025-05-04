import pickle
from functools import partial
from pathlib import Path
from timeit import default_timer as timer

import arviz as az
import hydra
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC
from omegaconf import DictConfig, OmegaConf

from benchmarks.disease_mapping.data import (
    get_grid,
    get_population,
    get_urban_rural,
)
from benchmarks.disease_mapping.samplers import (
    get_np_sampler,
    sample_gp,
    sample_gp_pointwise,
    sample_matheron,
    sample_prevalence,
)
from benchmarks.disease_mapping.utils import batch, map_fn, unbatch
from benchmarks.disease_mapping.visualize import plot_distribution, plot_predictions
from dl4bi.core.train import load_ckpt
from dl4bi.core.utils import breakpoint_if_nonfinite


def predict(seed, model, s_c, samples_c, s_t, x_t, batch_size):
    """
    Transforms samples $y_c (+params) | (s, x, np, n)_c$ into $y_t, theta_t | (s, x)_t, (s, np, n)_c$.
    """
    rng = jax.random.key(seed)

    y_c = samples_c["y"]
    params = samples_c

    print("Num samples:", y_c.shape[0])
    print("Num context locations:", s_c.shape[0])
    print("Num target locations:", s_t.shape[0])

    print("Using params:", {k: v.shape for k, v in params.items()})
    print("and inferred spatial effect y_c:", y_c.shape)

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
    elif model.lower() == "matheron":
        print("Using Matheron's rule for predictions.")
        sample_y_t = sample_matheron
        model_name = "matheron"
    else:
        model = Path(model)
        state, cfg_model = load_ckpt(model)
        model_name = model.stem
        print(f"Using {model_name} for predictions.")
        sample_y_t = get_np_sampler(state)  # already batched

    rng_y_t, rng_theta_t = jax.random.split(rng)

    sample_y_t = partial(sample_y_t, s_c=s_c, s_t=s_t)  # locations are fixed
    y_t = map_fn(sample_y_t)(
        rng=jax.random.split(rng_y_t, num_iters),
        y_c=y_c,
        **params,
    )

    if x_t is not None:
        D = x_t.shape[-1]
        x_t = jnp.broadcast_to(x_t, (*y_t.shape, D))

    z_t = map_fn(sample_prevalence)(
        rng=jax.random.split(rng_theta_t, num_iters),
        y=y_t,
        x=x_t,
        **params,
    )

    theta_t = jax.nn.sigmoid(z_t)

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
    samples_c = mcmc.get_samples()
    # Predict
    s_t = get_grid(cfg.iso, cfg.region, cfg.res)
    if "x" in data.keys():
        print("Using urban/rural covariate.")
        x_t = get_urban_rural(cfg.iso, s_t, cfg.year)
    else:
        print("Not using urban/rural covariate.")
        x_t = None

    print("Running predictions...")
    t_start = timer()
    model_name, s_t, y_t, theta_t = predict(
        cfg.seed,
        cfg.np,
        data["s"],
        samples_c,
        s_t,
        x_t,
        cfg.batch_size,
    )
    theta_t = theta_t.block_until_ready()
    t_end = timer()
    print(f"Took {t_end - t_start:.2f} seconds.")

    # Save results
    print("Saving results...")
    results_path = mcmc_results_path / model_name
    results_path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, results_path / "config.yaml")

    jnp.savez(results_path / "predictions.npz", s=s_t, theta=theta_t, y=y_t)

    # Plotting
    print("Plotting...")
    fig = plot_predictions(s_t, y_t, theta_t, data, samples_c["y"])
    fig.savefig(results_path / "predictions.png", dpi=300)

    # Aggregate results
    print("Aggregating over the area...")
    populations = get_population(cfg.iso, s_t, cfg.year, cfg.res)
    aggregate_samples = aggregate(theta_t, populations)

    jnp.save(results_path / "aggregate_samples.npy", aggregate_samples)
    fig = plot_distribution(aggregate_samples)
    fig.savefig(results_path / "aggregate_distribution.png", dpi=300)

    # Summary
    print("Preparing Arviz summary...")
    num_chains = mcmc.num_chains

    def split_chains(samples: jax.Array):
        return samples.reshape(num_chains, -1, *samples.shape[1:])

    posterior = {
        "aggregate_theta_t": aggregate_samples,
        "theta_t": theta_t,
        "y_t": y_t,
    }  # | samples
    # drop mcmc samples here for clarity

    observed_data = {"s_t": s_t, "x_t": x_t} | data

    az_data = az.from_dict(
        posterior=jax.tree.map(split_chains, posterior),
        observed_data=observed_data,
    )
    summary = az.summary(az_data, hdi_prob=0.95)
    print(summary)
    summary.to_csv(results_path / "summary.csv")
    print("Done")


if __name__ == "__main__":
    main()
