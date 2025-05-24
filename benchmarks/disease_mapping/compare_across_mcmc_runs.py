from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import pandas as pd
from jax import jit, vmap
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


def mmd(x, y):
    # MMD^2 with RBF 1 kernel
    assert x.ndim == y.ndim == 1

    def k(s, t):
        return jnp.exp(-((s - t) ** 2))

    mask_x = ~jnp.identity(x.shape[0], dtype=bool)
    kxx = jnp.mean(k(x[:, None], x[None, :]), where=mask_x)
    mask_y = ~jnp.identity(y.shape[0], dtype=bool)
    kyy = jnp.mean(k(y[:, None], y[None, :]), where=mask_y)
    kxy = jnp.mean(k(x[:, None], y[None, :]))
    return kxx + kyy - 2 * kxy


mmd = jit(vmap(mmd, in_axes=(1, 1)))


def evaluate(
    true_dist: jax.Array,  # [N, L]
    predicted_dist: jax.Array | None = None,  # [N, L]
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
    # metrics["MRE(mean)"] = mre(true_mean, predicted_mean)
    metrics["MAE(std)"] = mae(predicted_std, true_std)
    # metrics["MRE(std)"] = mre(true_std, predicted_std)
    metrics["RMSE(mean)"] = rmse(predicted_mean, true_mean)
    metrics["RMSE(std)"] = rmse(predicted_std, true_std)
    metrics["avg L2(map)"] = rmse(predicted_mean, true_dist)
    metrics["mean MMD2"] = mmd(predicted_dist, true_dist).mean()
    # this assumes diagonal normal posterior predictive
    metrics["avg pointwise NLL"] = -jsp.stats.norm.logpdf(
        true_dist, predicted_mean, predicted_std
    ).mean()

    z = jsp.stats.norm.ppf(0.975)
    pointwise_coverage = (true_dist >= predicted_mean - z * predicted_std) & (
        true_dist <= predicted_mean + z * predicted_std
    )
    metrics["avg coverage of predictive 0.95 hdi"] = jnp.mean(pointwise_coverage)

    # TODO: metrics when the predictive distribution is given by samples
    return metrics


def main(runs: list[Path], quantity: Literal["theta", "z"]):
    true_run = runs[0]
    rng = jax.random.key(0)
    candidates = ["matheron", "gp", "gp_pointwise"]
    for candidate in candidates:
        if (candidate_path := true_run / candidate).exists():
            true_dist = jnp.load(candidate_path / "predictions.npz")[quantity]
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
        for outputs_path in run.glob("**/predictions.npz"):
            outputs_path = outputs_path.parent
            if outputs_path == candidate_path:
                continue
            predicted_dist = jnp.load(outputs_path / "predictions.npz")
            print(predicted_dist)
            if "theta" in predicted_dist:
                # its from an MCMC run
                predicted_dist = predicted_dist[quantity]
                metrics = evaluate(true_dist, predicted_dist)
                metrics["model"] = outputs_path.name
                metrics |= run_params
                results.append(metrics)

                # per-chain statistics
                predicted_dist = predicted_dist.reshape(
                    num_chains, -1, *predicted_dist.shape[1:]
                )
                for i in range(num_chains):
                    metrics = evaluate(true_dist, predicted_dist[i])
                    metrics["model"] = outputs_path.name + f"/{i}"
                    metrics |= run_params | {"chains": [i]}
                    results.append(metrics)
            else:
                # its from an NP
                rng_i, rng = jax.random.split(rng)
                m, std = predicted_dist["mean"], predicted_dist["std"]
                predicted_dist = (
                    jax.random.normal(rng_i, shape=(4000, *m.shape)) * std + m
                )
                model_cfg = OmegaConf.load(outputs_path / "config.yaml")
                if model_cfg.output_format == "z" and quantity == "theta":
                    # convert to theta
                    predicted_dist = jax.nn.sigmoid(predicted_dist)
                elif model_cfg.output_format == "theta" and quantity == "z":
                    # convert to z
                    predicted_dist = jnp.clip(predicted_dist, 0, 1)
                    predicted_dist = jax.scipy.special.logit(predicted_dist).clip(
                        -1e6, 1e6
                    )

                metrics = evaluate(true_dist, predicted_dist)
                metrics["model"] = outputs_path.name
                metrics["seed"] = model_cfg.seed
                results.append(metrics)

    # constant predictor
    data = jnp.load(true_run / "data.npz")
    trivial_predictor = jnp.mean(data["n_pos"] / data["n"], axis=0)
    trivial_predictor = jnp.broadcast_to(trivial_predictor, true_dist.shape)
    results.append(
        {"model": "emprirical mean constant predictor"}
        | evaluate(true_dist, trivial_predictor)
    )

    df: pd.DataFrame = pd.DataFrame.from_records(results, index="model")
    df = df.groupby(["model"]).agg(["mean", "sem"]).round(3)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare MCMC runs.")
    parser.add_argument("runs", type=Path, nargs="+", help="Paths to MCMC runs.")
    parser.add_argument(
        "--theta", action="store_true", help="Use theta predictions instead of z."
    )
    args = parser.parse_args()
    if args.theta:
        quantity = "theta"
    else:
        quantity = "z"

    result = main(args.runs, quantity)

    print(result)
    result.to_csv("results.csv")
