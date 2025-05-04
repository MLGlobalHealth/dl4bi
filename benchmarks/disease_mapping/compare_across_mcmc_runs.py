from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm


def evaluate(
    true_dist: NDArray,  # [N, L]
    predicted_dist: NDArray | None = None,  # [N, L]
    predicted_mean: NDArray | None = None,  # [L]
    predicted_std: NDArray | None = None,  # [L]
):
    true_mean = np.mean(true_dist, axis=0)
    true_std = np.std(true_dist, axis=0)

    if predicted_dist is not None:
        predicted_mean = np.mean(predicted_dist, axis=0)
        predicted_std = np.std(predicted_dist, axis=0)
    assert predicted_mean is not None
    assert predicted_std is not None

    metrics = {}

    metrics["MEAN(true mean)"] = np.mean(true_mean)
    metrics["MAE(mean)"] = np.mean(np.abs(predicted_mean - true_mean))
    metrics["MAE(std)"] = np.mean(np.abs(predicted_std - true_std))
    metrics["RMSE(mean)"] = np.sqrt(np.mean((predicted_mean - true_mean) ** 2))
    metrics["RMSE(std)"] = np.sqrt(np.mean((predicted_std - true_std) ** 2))
    metrics["avg L2(map)"] = np.mean((true_dist - predicted_mean) ** 2)

    z = norm.ppf(0.975)
    pointwise_coverage = (true_dist >= predicted_mean - z * predicted_std) & (
        true_dist <= predicted_mean + z * predicted_std
    )
    metrics["avg coverage of predictive 0.95 hdi"] = np.mean(pointwise_coverage)

    # TODO: metrics when the predictive distribution is given by samples
    return metrics


def main(runs: list[Path]):
    true_run = runs[0]
    candidates = ["matheron", "gp", "gp_pointwise"]
    for candidate in candidates:
        if (candidate_path := true_run / candidate).exists():
            true_dist = np.load(candidate_path / "predictions.npz")["theta"]
            print(f"Using {candidate_path} as ground truth.")
            break
    else:
        raise ValueError("Not ground truth found.")

    data = []
    for run in runs:
        for outputs_path in run.glob("*/predictions.npz"):
            predicted_dist = np.load(outputs_path)
            print(predicted_dist)
            if "theta" in predicted_dist:
                predicted_dist = predicted_dist["theta"]
                metrics = evaluate(true_dist, predicted_dist)
            else:
                metrics = evaluate(
                    true_dist, None, predicted_dist["mean"], predicted_dist["std"]
                )
            metrics["model"] = run.name + "/" + outputs_path.parent.name
            data.append(metrics)

    df = pd.DataFrame.from_records(data, index="model")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare MCMC runs.")
    parser.add_argument("runs", type=Path, nargs="+", help="Paths to MCMC runs.")
    args = parser.parse_args()

    result = main(args.runs)

    print(result)
    result.to_csv("results.csv")
