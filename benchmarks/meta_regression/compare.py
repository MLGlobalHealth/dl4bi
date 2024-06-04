#!/usr/bin/env python3
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def compare(args):
    np.random.seed(args.seed)
    num_batches = 5000
    batch_idxs = np.random.choice(
        num_batches,
        size=args.num_samples,
        replace=False,
    )
    d = defaultdict(dict)
    for p in Path(args.directory).glob("*.pkl"):
        with open(p, "rb") as f:
            batch = pickle.load(f)
            for idx in batch_idxs:
                data, preds = batch[idx]
                d[idx]["data"] = data
                d[idx][p.stem] = preds
    for idx, dd in d.items():
        plot_posterior_predictive(args.directory, idx, dd)


def plot_posterior_predictive(directory: str, batch_id: int, dd: dict, hdi_prob=0.95):
    """Plots the posterior predictive alongside true values."""
    s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, var, ls = dd.pop(
        "data"
    )
    i = np.random.randint(s_ctx.shape[0])
    valid_len_ctx, valid_len_test = valid_lens_ctx[i], valid_lens_test[i]
    s_ctx, f_ctx = s_ctx[i, :valid_len_ctx, 0], f_ctx[i, :valid_len_ctx, 0]
    s_test, f_test = s_test[i, :valid_len_test, 0], f_test[i, :valid_len_test, 0]
    idx = np.argsort(s_test)
    s_test, f_test = s_test[idx], f_test[idx]
    z_score = np.abs(norm.ppf((1 - hdi_prob) / 2))
    fig, axes = plt.subplots(1, len(dd), sharey=True)
    if not isinstance(axes, list):
        axes = [axes]
    for ax, (name, preds) in zip(axes, dd.items()):
        f_mu, f_std = preds
        f_mu = f_mu[i, :valid_len_test, 0]
        if f_mu.shape == f_std.shape:  # diagonal covariance
            f_std = f_std[i, :valid_lens_test, 0]
        else:  # lower cholesky covariance, i.e. Sigma=LL^T
            f_std = np.diag(f_std[i] @ f_std[i].T)
        f_lower, f_upper = f_mu - z_score * f_std, f_mu + z_score * f_std
        ax.plot(s_test, f_test, color="black")
        ax.plot(s_test, f_mu[idx], color="steelblue")
        ax.scatter(s_ctx, f_ctx, color="black")
        ax.fill_between(
            s_test,
            f_lower[idx],
            f_upper[idx],
            alpha=0.4,
            color="steelblue",
            interpolate=True,
        )
        ax = plt.gca()
        ax.set_title(name.upper())
    title = f"Batch {batch_id} Sample {i} (var: {var[0]:0.2f}, ls: {ls[0]:0.2f})"
    plt.ylabel("f")
    plt.xlabel("s")
    plt.suptitle(title)
    path = Path(directory) / f"{title}.png"
    plt.savefig(path, dpi=150)
    plt.clf()
    plt.close()
    return path


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--directory",
        default="results/1D_GP/rbf",
        help="Directory with pkl files of results.",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=25,
        help="Number of samples to compare.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    compare(args)
