#!/usr/bin/env python3
import argparse
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, ttest_rel

from dsp.core import mvn_logpdf
from dsp.core.utils import mask_from_valid_lens


def plot_diff(args):
    dir = Path(args.directory)
    m1, m2 = args.model_1, args.model_2
    nlls, data = {m1: [], m2: []}, {m1: [], m2: []}
    keys = [
        "s_ctx",
        "f_ctx",
        "valid_lens_ctx",
        "s_test",
        "f_test",
        "valid_lens_test",
        "var",
        "ls",
    ]
    for m in [m1, m2]:
        with open(dir / f"{m}-seed-{args.seed}.pkl", "rb") as f:
            for tpl in pickle.load(f):
                batch, preds = tpl
                d = dict(zip(keys, batch))
                d["f_mu"], d["f_std"] = preds
                if d["f_mu"].shape == d["f_std"].shape:
                    L_test = d["s_test"].shape[1]
                    mask = mask_from_valid_lens(L_test, d["valid_lens_test"])
                    nll = -norm.logpdf(d["f_test"], d["f_mu"], d["f_std"]).mean(
                        where=mask
                    )
                else:
                    B = d["f_test"].shape[0]
                    mvn_nlls = -mvn_logpdf(
                        d["f_test"].reshape(B, -1),
                        d["f_mu"].reshape(B, -1),
                        d["f_std"],
                        is_tril=True,
                    )
                    nll = (mvn_nlls / d["valid_lens_test"]).mean()
                nlls[m] += [nll]
                data[m] += [d]
    nlls_m1 = np.hstack(nlls[m1])
    nlls_m2 = np.hstack(nlls[m2])
    diff = nlls_m1 - nlls_m2
    p_value = ttest_rel(nlls_m1, nlls_m2).pvalue
    print(f"Mean diff: {diff.mean():0.3f}, paired t-test p-value: {p_value:0.3f}")
    n_idxs = np.argsort(diff)[: args.worst_n]
    batch_size = data[m1][0]["s_test"].shape[0]
    batch_idxs, sample_idxs = n_idxs // batch_size, n_idxs % batch_size
    for i, (n_idx, b_idx, s_idx) in enumerate(zip(n_idxs, batch_idxs, sample_idxs)):
        d1, d2 = data[m1][b_idx], data[m2][b_idx]
        d1 = {k: v.squeeze() for k, v in d1.items()}
        d2 = {k: v.squeeze() for k, v in d2.items()}
        vc, vt = d1["valid_lens_ctx"][s_idx], d1["valid_lens_test"][s_idx]
        var, ls = d1["var"], d1["ls"]
        s_ctx, f_ctx = d1["s_ctx"][s_idx, :vc], d1["f_ctx"][s_idx, :vc]
        s_test, f_test = d1["s_test"][s_idx, :vt], d1["f_test"][s_idx, :vt]
        # TODO(danj): fix NLL calculation here
        f_mu_1, f_mu_2 = d1["f_mu"][s_idx, :vt], d2["f_mu"][s_idx, :vt]
        f_std_1, f_std_2 = d1["f_std"][s_idx], d2["f_std"][s_idx]
        f_std_1 = np.diag(f_std_1[:vt, :vt]) if f_std_1.ndim > 1 else f_std_1[:vt]
        f_std_2 = np.diag(f_std_2[:vt, :vt]) if f_std_2.ndim > 1 else f_std_2[:vt]
        title = f"Diff {i+1}: Batch {b_idx}, Sample {s_idx}"
        title += f", var: {var:0.2f}, ls {ls:0.2f}"
        title += f", NLL({m1})-NLL({m2})={diff[n_idx]:0.3f}"
        path = dir / f"{title}.pdf"
        plot_pp_comparison(
            path,
            title,
            m1,
            m2,
            s_ctx,
            f_ctx,
            s_test,
            f_test,
            f_mu_1,
            f_std_1,
            f_mu_2,
            f_std_2,
        )


def plot_pp_comparison(
    path: Path,
    title: str,
    model_1_name: str,
    model_2_name: str,
    s_ctx: np.ndarray,
    f_ctx: np.ndarray,
    s_test: np.ndarray,
    f_test: np.ndarray,
    f_mu_1: np.ndarray,
    f_std_1: np.ndarray,
    f_mu_2: np.ndarray,
    f_std_2: np.ndarray,
    hdi_prob=0.95,
):
    if f_mu_1.shape != f_std_1.shape:  # L
        f_std_1 = np.diag(f_std_1)
    if f_mu_2.shape != f_std_2.shape:  # L
        f_std_2 = np.diag(f_std_2)
    idx = np.argsort(s_test)
    z = np.abs(norm.ppf((1 - hdi_prob) / 2))
    f_lower_1, f_upper_1 = f_mu_1 - z * f_std_1, f_mu_1 + z * f_std_1
    f_lower_2, f_upper_2 = f_mu_2 - z * f_std_2, f_mu_2 + z * f_std_2
    plt.plot(s_test[idx], f_test[idx], color="black")
    plt.plot(s_test[idx], f_mu_1[idx], color="steelblue", label=model_1_name)
    plt.plot(s_test[idx], f_mu_2[idx], color="darkorange", label=model_2_name)
    plt.scatter(s_ctx, f_ctx, color="black")
    plt.fill_between(
        s_test[idx],
        f_lower_1[idx],
        f_upper_1[idx],
        alpha=0.5,
        color="steelblue",
        interpolate=True,
    )
    plt.fill_between(
        s_test[idx],
        f_lower_2[idx],
        f_upper_2[idx],
        alpha=0.25,
        color="darkorange",
        interpolate=True,
    )
    plt.ylabel("f")
    plt.xlabel("s")
    plt.legend()
    plt.title(title)
    plt.gcf().set_size_inches(10, 5)
    plt.savefig(path, dpi=150)
    plt.clf()
    plt.close()


def plot_random(args):
    np.random.seed(args.seed)
    num_batches = 5000
    batch_idxs = np.random.choice(
        num_batches,
        size=args.num_samples,
        replace=False,
    )
    d = defaultdict(dict)
    for p in Path(args.directory).glob("*.pkl"):
        if not re.match("|".join(args.only), p.stem):
            continue
        with open(p, "rb") as f:
            batches = pickle.load(f)
            for idx in batch_idxs:
                data, preds = batches[idx]
                d[idx]["data"] = data
                d[idx][p.stem] = preds
    for idx, dd in d.items():
        plot_posterior_predictive(args.directory, idx, dd)


def plot_posterior_predictive(
    directory: str,
    batch_id: int,
    dd: dict,
    hdi_prob=0.95,
):
    """Plots the posterior predictive alongside true values."""
    s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, var, ls = dd.pop(
        "data"
    )
    i = np.random.randint(s_ctx.shape[0])
    L_test = s_test.shape[1]
    valid_len_ctx, valid_len_test = valid_lens_ctx[i], valid_lens_test[i]
    s_ctx, f_ctx = s_ctx[i, :valid_len_ctx, 0], f_ctx[i, :valid_len_ctx, 0]
    s_test, f_test = s_test[i, :valid_len_test, 0], f_test[i, :valid_len_test, 0]
    idx = np.argsort(s_test)
    s_test, f_test = s_test[idx], f_test[idx]
    z_score = np.abs(norm.ppf((1 - hdi_prob) / 2))
    num_subplots = len(dd)
    fig, axes = plt.subplots(num_subplots, 1, sharex=True, constrained_layout=True)
    for ax, (name, preds) in zip(axes, sorted(dd.items())):
        f_mu, f_std = preds
        if f_mu[i].shape == f_std[i].shape:  # f_std is independent/diagonal
            f_mu = f_mu[i, :valid_len_test, 0]
            f_std = f_std[i, :valid_len_test, 0]
            nll = -norm.logpdf(f_test, f_mu, f_std).mean()
        else:  # f_std is a lower triangular covariance matrix
            f_mu = f_mu[i, :valid_len_test, 0]
            f_std = f_std[i, :valid_len_test, :valid_len_test]
            nll = -mvn_logpdf(
                f_test[None, ...], f_mu[None, ...], f_std[None, ...], is_tril=True
            ).mean()
            nll = nll / L_test  # average over L_test
            f_std = np.diag(f_std @ f_std.T)
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
        ax.set_title(f"{name.upper()}, NLL: {nll:0.3f}")
    title = (
        f"Random: Batch {batch_id} Sample {i} (var: {var[0]:0.2f}, ls: {ls[0]:0.2f})"
    )
    fig.set_size_inches(10, num_subplots * 5)
    fig.suptitle(title)
    plt.ylabel("f")
    plt.xlabel("s")
    path = Path(directory) / f"{title}.pdf"
    plt.savefig(path, dpi=150)
    plt.clf()
    plt.close()
    return path


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub_p = parser.add_subparsers(dest="cmd")
    diff_p = sub_p.add_parser("diff")
    parser.add_argument(
        "-d",
        "--directory",
        default="results/1D_GP/rbf",
        help="Directory with pkl files of results.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=7,
        help="Random seed.",
    )
    diff_p.add_argument(
        "model_1",
        type=str,
        default="tnpd",
        help="Model 1 for comparison.",
    )
    diff_p.add_argument(
        "model_2",
        type=str,
        default="sptx_fast",
        help="Model 2 for comparison.",
    )
    diff_p.add_argument(
        "-n",
        "--worst_n",
        type=int,
        default=10,
        help="Plot worst n differences.",
    )
    random_p = sub_p.add_parser("random")
    random_p.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=25,
        help="Number of samples to compare.",
    )
    random_p.add_argument(
        "-o",
        "--only",
        type=str,
        nargs="*",
        default=[],
        help="Restrict evaluation to models in this list.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    match args.cmd:
        case "diff":
            plot_diff(args)
        case "random":
            plot_random(args)
        case _:
            pass
