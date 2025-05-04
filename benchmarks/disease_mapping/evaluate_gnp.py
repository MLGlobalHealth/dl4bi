from pathlib import Path
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pandas as pd
from jax import jit
from omegaconf import OmegaConf
from scipy.stats import chi2  # jax doesn't implement ppf

from benchmarks.disease_mapping.data import get_urban_rural
from benchmarks.disease_mapping.visualize import map_grid, scatter_map
from dl4bi.core.model_output import DiagonalMVNOutput
from dl4bi.core.train import load_ckpt


def plot_side_by_side(s, gp_mean, gp_std, predicted_mean, predicted_std):
    fig, axes = map_grid(s, 2, 2)
    fig.suptitle("GP vs Neural Process")

    std_min = min(gp_std.min(), predicted_std.min())
    std_max = max(gp_std.max(), predicted_std.max())
    mean_min = 0
    mean_max = 1

    scatter_map(s, gp_mean, ax=axes[0, 0], vmin=mean_min, vmax=mean_max)
    scatter_map(s, gp_std, ax=axes[0, 1], vmin=std_min, vmax=std_max)
    scatter_map(s, predicted_mean, ax=axes[1, 0], vmin=mean_min, vmax=mean_max)
    scatter_map(s, predicted_std, ax=axes[1, 1], vmin=std_min, vmax=std_max)
    axes[0, 0].set_title("GP mean")
    axes[0, 1].set_title("GP std")
    axes[1, 0].set_title("NP mean")
    axes[1, 1].set_title("NP std")
    return fig


def rmse(x, y):
    return jnp.sqrt(jnp.mean((x - y) ** 2))


def mae(x, y):
    return jnp.mean(jnp.abs(x - y))


def mvn_hdi(mean, cov, samples):
    # assert last dimension is 1 everywhere
    # check how many samples from x are in the 95%-highest-density-region of a multivariate normal
    # https://stats.stackexchange.com/questions/354063/computing-highest-density-region-given-multivariate-normal-distribution-with-dim

    d = samples.shape[-1]
    x = samples - mean
    y = chi2(d).ppf(0.95)
    if cov.shape == (d,):  # diagonal
        return jnp.mean(jnp.sum(x**2 / cov, axis=-1) <= y)
    elif cov.shape == (d, d):  # full covariance
        x = x[..., None]
        return jnp.mean(x.mT @ jsp.linalg.solve(cov, x, assume_a="pos") <= y)
    else:
        raise ValueError(f"Invalid covariance shape: {cov.shape}")


def evaluate(mcmc_path: Path, gnp_path: Path):
    rng = jax.random.key(0)

    if isinstance(mcmc_path, str):
        mcmc_path = Path(mcmc_path)

    true_models = ["gp", "gp_pointwise"]  # ground truth models in order of preference

    for true_model in true_models:
        path = mcmc_path / true_model
        if path.exists():
            print(f"Using {true_model} as ground truth.")
            true_dist = dict(jnp.load(path / "predictions.npz"))
            true_cfg = OmegaConf.load(path / "config.yaml")
            break
    else:
        raise ValueError(
            f"No ground truth found. One of {true_models} must be in the mcmc path."
        )

    # Load data
    data = jnp.load(mcmc_path / "data.npz")
    print(data.keys())

    s_c, n, n_pos = data["s"], data["n"], data["n_pos"]
    s_t, theta_t = true_dist["s"], true_dist["theta"]

    if "x" in data:
        x_c = data["x"]
        x_t = get_urban_rural(true_cfg.iso, s_t, true_cfg.year)
        print("Using x.")
    else:
        x_c = x_t = None
        print("No x.")

    # Load GNP model
    state, model_cfg = load_ckpt(gnp_path)
    model_name = model_cfg.name
    match model_cfg.get("input_format", "survey"):
        case "survey":
            y_c = jnp.stack([n_pos, n], axis=-1)
        case "theta":
            y_c = (n_pos / n)[..., None]

    @jit
    def predict(rng, s_c, x_c, y_c, s_t, x_t):
        # expects a single task, not a batch
        output = state.apply_fn(
            {"params": state.params, **state.kwargs},
            x_ctx=x_c[None] if x_c is not None else None,
            s_ctx=s_c[None],
            f_ctx=y_c[None],
            x_test=x_t[None] if x_t is not None else None,
            s_test=s_t[None],
            rngs={"extra": rng},
        )
        match output:
            case DiagonalMVNOutput(mean, std):
                # coerce shape to [L]
                return jnp.squeeze(mean, axis=[0, 2]), jnp.squeeze(std, axis=[0, 2])
            case _:
                raise NotImplementedError()

    print("Compiling model...")
    rng_c = jax.random.key(0)
    time_start = timer()
    jax.block_until_ready(
        predict(
            rng_c,
            jax.random.uniform(rng_c, s_c.shape),
            jax.random.uniform(rng_c, x_c.shape) if x_c is not None else None,
            jax.random.uniform(rng_c, y_c.shape),
            jax.random.uniform(rng_c, s_t.shape),
            jax.random.uniform(rng_c, x_t.shape) if x_t is not None else None,
        )
    )
    time_end = timer()
    print(f"Took {time_end - time_start:.2f} seconds.")
    print("Running predictions...")
    # mean, std of dimensions [L]
    time_start = timer()
    predicted_mean, predicted_std = jax.block_until_ready(
        predict(rng, s_c, x_c, y_c, s_t, x_t)
    )
    time_end = timer()
    print(f"Took {time_end - time_start:.2f} seconds.")
    print("Saving predictions...")
    results_path = mcmc_path / model_name
    results_path.mkdir(parents=True, exist_ok=True)
    jnp.savez(
        results_path / "predictions.npz",
        s=s_t,
        mean=predicted_mean,
        std=predicted_std,
    )
    # Plotting
    fig = plot_side_by_side(
        s_t,
        theta_t.mean(0),
        theta_t.std(0),
        predicted_mean,
        predicted_std,
    )
    fig.savefig(results_path / "predictions.png", dpi=300)

    # ---
    # Benchmarks
    # ---
    print("Running benchmarks...")
    true_samples = theta_t
    true_mean = true_samples.mean(axis=0)
    true_std = true_samples.std(axis=0)

    ci = 0.95
    z_score = jnp.abs(jax.scipy.stats.norm.ppf((1 - ci) / 2))
    lo = predicted_mean - z_score * predicted_std
    up = predicted_mean + z_score * predicted_std

    results = dict()
    # pointwise-averaged metrics
    results["true mean"] = true_mean.mean()  # to provide context for rmse
    results["rmse mean"] = rmse(true_mean, predicted_mean)
    results["MAP L2 loss"] = (
        (true_samples - predicted_mean) ** 2
    ).mean()  # L2 loss averaged over samples and locations
    results["rmse std"] = rmse(true_std, predicted_std)
    results["average coverage"] = jnp.mean((lo <= true_samples) & (true_samples <= up))

    # global metrics
    # this nll is normalized by num samples and num locations
    results["data nll"] = -jsp.stats.norm.logpdf(
        true_samples, predicted_mean, predicted_std
    ).mean()
    results["coverage"] = mvn_hdi(predicted_mean, predicted_std, true_samples)

    df = pd.DataFrame(results.items(), columns=["metric", "value"])
    df.to_csv(results_path / "metrics.csv", index=False)
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "gnp_path", type=Path, help="Path to the generalized neural process model."
    )
    parser.add_argument(
        "mcmc_path",
        type=Path,
        help="Path to the directory containing MCMC samples.",
        default=None,
        nargs="*",
    )
    args = parser.parse_args()

    mcmc_path = (
        args.mcmc_path[0]
        if args.mcmc_path
        else max(Path("results").glob("MCMC*"), key=lambda p: p.stat().st_mtime)
    )
    print(f"Using MCMC path: {mcmc_path}")

    results = evaluate(mcmc_path, args.gnp_path)
    print(results)


if __name__ == "__main__":
    main()
