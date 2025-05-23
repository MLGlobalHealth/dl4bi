from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import pandas as pd
from jax import scipy as jsp
from omegaconf import OmegaConf


def mse(x, y):
    return jnp.mean((x - y) ** 2)


def rmse(x, y):
    return jnp.sqrt(mse(x, y))


def mae(x, y):
    return jnp.mean(jnp.abs(x - y))


def mre(x, y):
    return jnp.mean(jnp.abs(x - y) / (jnp.abs(x) + 1e-8))


def evaluate(
    true_dist: jax.Array,  # [N, L]
    predicted_dist: jax.Array | None = None,  # [N, L]
    predicted_mean: jax.Array | None = None,  # [L]
    predicted_std: jax.Array | None = None,  # [L]
    predicted_cov: jax.Array | None = None,  # [L, L]
):
    true_mean = jnp.mean(true_dist, axis=0)
    true_std = jnp.std(true_dist, axis=0)

    if predicted_dist is not None:
        predicted_mean = jnp.mean(predicted_dist, axis=0)
        predicted_std = jnp.std(predicted_dist, axis=0)
        # predicted_cov = jnp.cov(predicted_dist, rowvar=False)

    metrics = {}
    metrics["N samples"] = predicted_dist.shape[0] if predicted_dist is not None else 0
    metrics["MEAN(true mean)"] = jnp.mean(true_mean)
    metrics["MAE(mean)"] = mae(predicted_mean, true_mean)
    metrics["MRE(mean)"] = mre(predicted_mean, true_mean)
    metrics["MAE(std)"] = mae(predicted_std, true_std)
    metrics["MRE(std)"] = mre(predicted_std, true_std)
    metrics["RMSE(mean)"] = rmse(predicted_mean, true_mean)
    metrics["RMSE(std)"] = rmse(predicted_std, true_std)
    metrics["avg L2(map)"] = rmse(predicted_mean, true_mean)
    # this assumes diagonal normal posterior predictive
    metrics["avg pointwise NLL"] = -jsp.stats.norm.logpdf(
        true_dist, predicted_mean, predicted_std
    ).mean()
    if predicted_cov is not None:
        metrics["avg NLL"] = -jsp.stats.multivariate_normal.logpdf(
            true_dist, predicted_mean, predicted_cov
        ).mean()

    z = jsp.stats.norm.ppf(0.975)
    pointwise_coverage = (true_dist >= predicted_mean - z * predicted_std) & (
        true_dist <= predicted_mean + z * predicted_std
    )
    metrics["avg coverage of predictive 0.95 hdi"] = jnp.mean(pointwise_coverage)

    # TODO: metrics when the predictive distribution is given by samples
    return metrics


def main(
    runs: list[Path],
    target_quantity: Literal["theta", "z"],
    seed=0,
    n_samples_for_conversion=1000,
):
    rng = jax.random.key(seed)
    true_run = runs[0]
    candidates = ["matheron", "gp", "gp_pointwise"]
    for candidate in candidates:
        if (candidate_path := true_run / candidate).exists():
            true_dist = jnp.load(candidate_path / "predictions.npz")[target_quantity]
            print(f"Using {candidate_path} as ground truth.")
            break
    else:
        raise ValueError("Not ground truth found.")

    results = []
    for run in runs:
        mcmc_config = OmegaConf.load(run / "config.yaml")
        num_chains = mcmc_config.num_chains
        run_params = {
            "chains": jnp.arange(num_chains),
            "num_samples": mcmc_config.num_samples,
            "num_warmup": mcmc_config.num_warmup,
        }
        for outputs_path in run.glob("*/predictions.npz"):
            predicted_dist = jnp.load(outputs_path)
            print(predicted_dist)
            if target_quantity in predicted_dist:
                # its from an MCMC run
                predicted_dist = predicted_dist[target_quantity]
                metrics = evaluate(true_dist, predicted_dist)
                metrics["model"] = outputs_path.parent.name
                metrics |= run_params
                results.append(metrics)

                # per-chain statistics
                predicted_dist = predicted_dist.reshape(
                    num_chains, -1, *predicted_dist.shape[1:]
                )
                for i in range(num_chains):
                    metrics = evaluate(true_dist, predicted_dist[i])
                    metrics["model"] = outputs_path.parent.name + f"/{i}"
                    metrics |= run_params | {"chains": [i]}
                    results.append(metrics)
            else:
                # its from a Neural Process
                mean, std = predicted_dist["mean"], predicted_dist["std"]
                output_format = OmegaConf.load(outputs_path.parent / "config.yaml").get(
                    "output_format", "theta"
                )
                match (output_format, target_quantity):
                    case ("theta", "theta") | ("z", "z"):
                        metrics = evaluate(true_dist, None, mean, std)
                    case ("z", "theta"):
                        metrics = evaluate(
                            true_dist,
                            jax.nn.sigmoid(
                                jax.random.normal(
                                    rng, (n_samples_for_conversion, *mean.shape)
                                )
                                * std
                                + mean
                            ),
                        )
                    case ("theta", "z"):
                        metrics = evaluate(
                            true_dist,
                            jsp.special.logit(
                                jax.random.normal(
                                    rng, (n_samples_for_conversion, *mean.shape)
                                )
                                * std
                                + mean
                            ),
                        )

                metrics["model"] = outputs_path.parent.name
                results.append(metrics)

    # constant predictor
    data = jnp.load(true_run / "data.npz")
    trivial_predictor = jnp.mean(data["n_pos"] / data["n"], axis=0)
    results.append(
        {"model": "emprirical mean constant predictor"}
        | evaluate(true_dist, None, trivial_predictor, 0)
    )

    df = pd.DataFrame.from_records(results, index="model")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare MCMC runs.")
    parser.add_argument("runs", type=Path, nargs="+", help="Paths to MCMC runs.")
    parser.add_argument(
        "--theta", action="store_true", help="Use theta as target quantity."
    )
    parser.add_argument("--z", action="store_true", help="Use z as target quantity.")
    args = parser.parse_args()

    if args.theta and args.z:
        raise ValueError("Cannot use both theta and z as target quantity.")
    elif args.z:
        target_quantity = "z"
    else:
        target_quantity = "theta"

    result = main(args.runs, target_quantity)

    print(result)
    result.to_csv("results.csv")
