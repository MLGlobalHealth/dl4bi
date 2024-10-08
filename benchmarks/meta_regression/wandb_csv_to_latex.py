#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import pandas as pd

# NOTE: first download the results you care about from wandb,
# e.g. name, kernel, seed, runtime, valid_loss, test_loss, etc
# then run this script to generate latex tables summarizing it
#
# Examples:
# ./wandb_csv_to_latex.py ~/scratch/gp_1d.csv -n "1D GP" -g kernel Name -p kernel
# ./wandb_csv_to_latex.py ~/scratch/celeba.csv -n "CelebA"
# ./wandb_csv_to_latex.py ~/scratch/mnist.csv -n "MNIST"


def main(args):
    df = pd.read_csv(args.path)
    func = lambda x: f"${np.mean(x):.3f}\\pm{np.std(x):0.3f}$"
    x = df[[*args.group_by, *args.metrics]].groupby(args.group_by).agg(func)
    x = x.reset_index()
    if args.pivot:
        index = args.group_by
        index.remove(args.pivot)
        x = x.pivot(index=index, columns=args.pivot, values=args.metrics)
        x = x.reset_index()
    x_tex = x.to_latex(
        index=False,
        caption=args.name,
        label=args.name.lower().replace(" ", "-"),
        float_format="{:0.4f}".format,
        column_format="l" + "r" * (len(x.columns) - 1),
    )
    print(x_tex)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", help="Path to csv.")
    parser.add_argument(
        "-m",
        "--metrics",
        nargs="+",
        default=["Test NLL", "Test Coverage", "Test RMSE", "Test MAE", "Runtime"],
        help="Metric columns to apply mean/std to.",
    )
    parser.add_argument(
        "-g",
        "--group_by",
        nargs="+",
        default=["Name", "kernel.kwargs.kernel.func"],
        help="Columns to group by.",
    )
    parser.add_argument(
        "-p",
        "--pivot",
        nargs="?",
        default="kernel.kwargs.kernel.func",
        help="Pivot to column headers.",
    )
    parser.add_argument(
        "-n",
        "--name",
        default="My Experiment",
        help="Name to use for caption and label.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args)
