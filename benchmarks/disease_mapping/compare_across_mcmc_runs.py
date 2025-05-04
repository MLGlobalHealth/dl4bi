from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd
from jax import scipy as jsp


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

    metrics["MEAN(true mean)"] = jnp.mean(true_mean)
    metrics["MAE(mean)"] = jnp.mean(jnp.abs(predicted_mean - true_mean))
    metrics["MAE(std)"] = jnp.mean(jnp.abs(predicted_std - true_std))
    metrics["RMSE(mean)"] = jnp.sqrt(jnp.mean((predicted_mean - true_mean) ** 2))
    metrics["RMSE(std)"] = jnp.sqrt(jnp.mean((predicted_std - true_std) ** 2))
    metrics["avg L2(map)"] = jnp.mean((true_dist - predicted_mean) ** 2)
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


def main(runs: list[Path]):
    true_run = runs[0]
    candidates = ["matheron", "gp", "gp_pointwise"]
    for candidate in candidates:
        if (candidate_path := true_run / candidate).exists():
            true_dist = jnp.load(candidate_path / "predictions.npz")["theta"]
            print(f"Using {candidate_path} as ground truth.")
            break
    else:
        raise ValueError("Not ground truth found.")

    results = []
    for run in runs:
        for outputs_path in run.glob("*/predictions.npz"):
            predicted_dist = jnp.load(outputs_path)
            print(predicted_dist)
            if "theta" in predicted_dist:
                predicted_dist = predicted_dist["theta"]
                metrics = evaluate(true_dist, predicted_dist)
            else:
                metrics = evaluate(
                    true_dist, None, predicted_dist["mean"], predicted_dist["std"]
                )
            metrics["model"] = run.name + "/" + outputs_path.parent.name
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
    args = parser.parse_args()

    result = main(args.runs)

    print(result)
    result.to_csv("results.csv")
