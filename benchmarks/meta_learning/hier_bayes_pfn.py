#!/usr/bin/env python
import argparse
import sys

import numpy as np
from tabpfn import TabPFNRegressor


def main(path):
    d = np.load(path, allow_pickle=True).item()
    reg = TabPFNRegressor(random_state=42)
    reg.fit(d["data"]["s_ctx"], d["data"]["f_ctx"])
    f_lower, f_mu, f_upper = reg.predict(
        d["data"]["s"],
        output_type="quantiles",
        quantiles=[0.025, 0.5, 0.975],
    )
    d["predictions"]["tabpfn"] = {"f_lower": f_lower, "f_mu": f_mu, "f_upper": f_upper}
    np.save(path, d)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", help="Path to <inference>.npy file.")
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args.path)
