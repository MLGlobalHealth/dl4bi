#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import sem

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
    if "Runtime" in df.columns:
        if args.runtime_numerator:
            df["Runtime"] = args.runtime_numerator / df.Runtime
        elif args.runtime_denominator:
            df["Runtime"] = df.Runtime / args.runtime_denominator
    if args.filter:
        df = df.query(args.filter)
    func = lambda x: f"${np.mean(x):.2f}\\pm{sem(x):0.2f}$"
    x = df[[*args.group_by, *args.metrics]].groupby(args.group_by).agg(func)
    x = x.reset_index()
    if args.pivot:
        index = [c for c in args.group_by if c != args.pivot]
        x = x.pivot(index=index, columns=args.pivot, values=args.metrics)
        x = x.reset_index()
    x_tex = x.to_latex(
        index=False,
        caption=args.name,
        label=args.name.lower().replace(" ", "-"),
        float_format="{:.2f}".format,
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
        default=["Test NLL", "Test MAE", "Test RMSE", "Test Coverage", "Runtime"],
        help="Metric columns to apply mean/std to.",
    )
    parser.add_argument(
        "-g",
        "--group_by",
        nargs="+",
        default=["Name"],
        help="Columns to group by.",
    )
    parser.add_argument(
        "-f",
        "--filter",
        help="A filter passed to pd.DataFrame.query before summarizing.",
        default="",
    )
    parser.add_argument(
        "-p",
        "--pivot",
        nargs="?",
        help="Pivot to column headers.",
    )
    parser.add_argument(
        "-n",
        "--name",
        default="My Experiment",
        help="Name to use for caption and label.",
    )
    parser.add_argument(
        "--runtime_numerator",
        type=float,
        default=None,
        help="Runtime = Numerator / Runtime.",
    )
    parser.add_argument(
        "--runtime_denominator",
        type=float,
        default=None,
        help="Runtime = Runtime / Denominator.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args)
