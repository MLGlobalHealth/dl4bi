from functools import partial
from itertools import chain
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
from jax import jit, vmap
from pandas import DataFrame

from benchmarks.disease_mapping.utils import make_pairwise


def l2(x: jax.Array, y: jax.Array):
    assert x.shape == y.shape
    return jnp.linalg.norm(x - y, ord=2)


def l2_squared(x: jax.Array, y: jax.Array):
    assert x.shape == y.shape
    return jnp.sum((x - y) ** 2)


@partial(jit, static_argnames=["kernel"])
def mmd(x: jax.Array, y: jax.Array, kernel: Callable):
    """
    MMD^2 estimate.
    Expects `x, y` of shapes `[N, sample_shape]`, `[M, sample_shape]`,
    and kernel taking arguments of shape `[sample_shape]`.

    See https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    """
    N, *sample_shape_x = x.shape
    M, *sample_shape_y = y.shape
    assert sample_shape_x == sample_shape_y

    # vectorized doesn't fit in memory for large sample sizes
    k = jit(make_pairwise(kernel, method="sequential"))

    k_xx: jax.Array = k(x, x)
    k_yy: jax.Array = k(y, y)
    k_xy: jax.Array = k(x, y)

    mask_x = ~jnp.eye(N, dtype=bool)
    mask_y = ~jnp.eye(M, dtype=bool)

    mmd2 = k_xx.mean(where=mask_x) - 2 * jnp.mean(k_xy) + jnp.mean(k_yy, where=mask_y)
    return mmd2


def mmd_gaussian(x: jax.Array, y: jax.Array):
    """MMD^2 with Gaussian kernel with `ls` = sample dimension."""

    def kernel(x1, x2):
        ls = x1.size  # normalize by dimension
        return jnp.exp(-l2_squared(x1, x2) / 2 / ls**2)

    return mmd(x, y, kernel)


def energy_distance(x: jax.Array, y: jax.Array):
    def kernel(x1, x2):
        return -l2(x1, x2)

    return mmd(x, y, kernel)


def wasserstein_distance(x: jax.Array, y: jax.Array):
    assert x.shape == y.shape
    assert x.ndim == 1

    x = jnp.sort(x)
    y = jnp.sort(y)

    return jnp.mean(jnp.abs(x - y))


def kolmogorov_smirnov(x: jax.Array, y: jax.Array):
    assert x.shape == y.shape
    (N,) = x.shape

    x = jnp.sort(x)
    y = jnp.sort(y)
    all = jnp.concat([x, y])

    cdf_x = jnp.searchsorted(x, all, side="right") / N
    cdf_y = jnp.searchsorted(y, all, side="right") / N
    return jnp.abs(cdf_x - cdf_y).max()


def rmse(x: jax.Array, y: jax.Array):
    return jnp.sqrt(jnp.mean((x - y) ** 2))


def mae(x: jax.Array, y: jax.Array):
    return jnp.mean(jnp.abs(x - y))


def coverage(x: jax.Array, y: jax.Array):
    """
    What fraction of x is covered by the central 95% CI of y.
    x, y of shape [N_samples, sample_shape]
    """
    q = jnp.array([2.5, 97.5])
    y_lo, y_up = jnp.percentile(y, q, axis=0)
    return jnp.mean((x >= y_lo) & (x <= y_up), axis=0)


@jit
def metrics(true, pred):
    """
    Args:
        true: samples from the true distribution of shape [N_samples, sample_shape]
        predicted: samples from the predicted distribution of shape [N_samples, sample_shape]
    Returns:
        a dictionary of metrics
    """
    assert true.shape == pred.shape
    N, *sample_shape = true.shape
    true = true.reshape(N, -1)
    pred = pred.reshape(N, -1)

    results = dict(num_samples=N, sample_shape=sample_shape)
    results["mmd"] = mmd_gaussian(true, pred)
    # NOTE: mmd doesn't make sense for the pointwise ground truth,
    # also in high dimensions the current implementation doesn't fit in memory
    results["rmse_pointwise_mean"] = rmse(
        jnp.mean(true, axis=0), jnp.mean(pred, axis=0)
    )
    results["rmse_pointwise_std"] = rmse(jnp.std(true, axis=0), jnp.std(pred, axis=0))

    results["mean_pointwise_coverage"] = coverage(true, pred).mean()
    results["mean_pointwise_coverage_reverse"] = coverage(pred, true).mean()
    results["mean_pointwise_wd"] = (vmap(wasserstein_distance, -1)(true, pred)).mean()

    return results


def thin(tree, factor: int):
    return jax.tree.map(lambda x: x[::factor], tree)


def evaluate(mcmc_path: Path | str, thinning=1):
    if isinstance(mcmc_path, str):
        mcmc_path = Path(mcmc_path)

    true_models = ["gp", "gp_pointwise"]  # ground truth models in order of preference

    for true_model in true_models:
        path = mcmc_path / true_model
        if path.exists():
            print(f"Using {true_model} as ground truth.")
            true_dist = dict(jnp.load(path / "predictions.npz"))
            true_dist = thin(true_dist, thinning)
            break
    else:
        raise ValueError(
            f"No ground truth found. One of {true_models} must be in the mcmc path."
        )

    results = {}
    for dir in mcmc_path.iterdir():
        if dir.is_dir():
            pred_model = dir.name

            print(f"Evaluating {pred_model} against {true_model}.")
            pred_dist = dict(jnp.load(dir / "predictions.npz"))
            pred_dist = thin(pred_dist, thinning)

            keys = true_dist.keys() & pred_dist.keys()
            keys -= {"s"}
            results[pred_model] = {k: metrics(true_dist[k], pred_dist[k]) for k in keys}

    results = jax.tree.map(lambda x: x.item(), results)

    records = [
        dict(pred_model=pred_model, true_model=true_model, variable=variable) | metrics
        for pred_model in results.keys()
        for variable, metrics in results[pred_model].items()
    ]

    df = DataFrame.from_records(records)

    # some column reordering since jax sorted the keys in metrics
    columns_first = ["pred_model", "true_model", "num_samples"]
    columns_last = ["sample_shape"]
    columns = df.columns.to_list()
    for col in chain(columns_first, columns_last):
        columns.remove(col)
    df = df[columns_first + columns + columns_last]

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mcmc_path",
        type=Path,
        help="Path to the directory containing MCMC samples.",
        default=None,
        nargs="*",
    )
    parser.add_argument("-t", "--thinning", type=int, default=1)
    args = parser.parse_args()

    mcmc_path = args.mcmc_path or max(
        Path("results").glob("MCMC_*"), key=lambda p: p.stat().st_mtime
    )
    print(f"Using MCMC path: {mcmc_path}")

    results = evaluate(mcmc_path, args.thinning)
    results.to_csv(mcmc_path / "metrics.csv")
    print(results)


if __name__ == "__main__":
    main()
