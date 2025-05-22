import argparse
import pickle
from pathlib import Path
from re import findall
from timeit import default_timer as timer

import arviz as az
import jax
import jax.numpy as jnp
import survey_model
from jax import jit, random, vmap
from numpyro.handlers import condition, seed
from numpyro.infer import MCMC
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from benchmarks.disease_mapping import data
from benchmarks.disease_mapping.samplers import sample_gp_pointwise_generic


def main(s, mcmc_path, batch_size=8, res=150):
    rng = random.key(s)
    mcmc_cfg: DictConfig = OmegaConf.load(mcmc_path / "config.yaml")
    with open(mcmc_path / "mcmc.pickle", "rb") as f:
        mcmc: MCMC = pickle.load(f)
    iso = mcmc_cfg.get("iso")
    (year,) = findall("\\d+", mcmc_cfg.query)
    region = None
    num_chains = mcmc.num_chains

    samples = mcmc.get_samples()
    s_c = jnp.load(mcmc_path / "data.npz")["s"]
    y_c = samples.pop("y")
    N = y_c.shape[0]
    s_t = data.get_grid(iso, region, res)
    x_t = data.get_urban_rural(iso, s_t, year) if mcmc_cfg.urban_rural else None

    print(s_c.shape, y_c.shape, s_t.shape, x_t.shape if x_t is not None else None)

    @jit
    def sampler(x):
        rng, y_c, params = x
        rng_y, rng_z = random.split(rng)

        y_t = sample_gp_pointwise_generic(
            rng_y,
            s_c,
            y_c,
            s_t,
            kernel=survey_model.kernel,
            method="map",  # will run preds in blocks of size 1024
            **params,
        )
        z_t = seed(condition(survey_model.prevalence, params), rng_z)(y_t, x_t)
        return y_t, z_t

    print(f"Running predictions with pointwise gp - batch size {batch_size}")
    start_time = timer()
    results = []
    assert N % batch_size == 0, f"Batch size {batch_size} does not divide {N}"
    for i in trange(
        N // batch_size, desc="Predicting", unit=" batches", dynamic_ncols=True
    ):
        rng_i, rng = random.split(rng)
        rng_i = random.split(rng_i, num_chains)
        y_c_i = y_c[i : i + batch_size]
        params_i = jax.tree.map(lambda x: x[i : i + batch_size], samples)
        pred = vmap(sampler)((rng_i, y_c_i, params_i))
        results.append(pred)
    y_t, z_t = zip(*results)
    y_t = jnp.concatenate(y_t, axis=0)
    z_t = jnp.concatenate(z_t, axis=0)
    z_t = jax.block_until_ready(z_t)
    end_time = timer()
    print(f"Prediction time: {end_time - start_time:.2f} seconds")
    theta_t = jax.nn.sigmoid(z_t)

    # Save predictions
    results_path = mcmc_path / "gp_pointwise"
    results_path.mkdir(parents=True, exist_ok=True)
    jnp.savez(
        results_path / "predictions.npz", s=s_t, y=y_t, theta=theta_t, z=z_t, x=x_t
    )
    (results_path / "time.txt").write_text(f"{end_time - start_time:.2f} seconds")

    # Generating arviz summary
    posterior = {"theta_t": theta_t, "z_t": z_t, "y_t": y_t}

    az_data = az.from_dict(
        jax.tree.map(lambda x: x.reshape(num_chains, -1, *x.shape[1:]), posterior)
    )
    summary = az.summary(az_data, hdi_prob=0.95)
    print(summary)
    summary.to_csv(results_path / "summary.csv")
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mcmc_path", type=Path, help="Path to the directory containing MCMC samples."
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=8, help="Batch size for predictions."
    )
    parser.add_argument("-s", "--seed", type=int, default=11, help="Random seed.")
    args = parser.parse_args()

    main(args.seed, args.mcmc_path, args.batch_size)
