#!/usr/bin/env python3
import argparse
import sys

import pandas as pd
from scipy.stats import ttest_rel


def paired_t_test(
    df: pd.DataFrame,
    model_1: str,
    model_2: str,
    statistic: str,
    filter: str,
    method: str = "less",
):
    # NOTE: this assumes that models appear in the same seed order
    if filter:
        df = df.query(filter)
    m1 = df[df.Name == model_1][statistic].values
    m2 = df[df.Name == model_2][statistic].values
    diff = m1 - m2
    print(f"{model_1}:\n{m1}")
    print(f"\n{model_2}:\n{m2}")
    print(f"\nDifferences:\n{diff}\n")
    return ttest_rel(m1, m2, alternative=method)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", help="Path to csv.")
    parser.add_argument("model_1", help="Model 1 name.")
    parser.add_argument("model_2", help="Model 2 name.")
    parser.add_argument(
        "-s",
        "--statistic",
        help="Statistic to compare.",
        default="Test NLL",
    )
    parser.add_argument(
        "-f",
        "--filter",
        help="A filter passed to pd.DataFrame.query before performing the t-test.",
        default="",
    )
    parser.add_argument(
        "-m",
        "--method",
        default="less",
        help="Which paired method.",
        choices=["less", "two-sided"],
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    df = pd.read_csv(args.path)
    result = paired_t_test(
        df,
        args.model_1,
        args.model_2,
        args.statistic,
        args.filter,
        args.method,
    )
    print(f"t-test ({args.method}) p-value: {result.pvalue:0.3f}")
