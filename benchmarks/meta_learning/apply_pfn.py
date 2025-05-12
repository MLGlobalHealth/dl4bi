#!/usr/bin/env -S PYENV_VERSION=torch python3
import argparse
import sys

import numpy as np
from tabpfn import TabPFNRegressor


def main(path):
    d = np.load(path, allow_pickle=True).item()
    x_train, x_test = d["x_train"], d["x_test"]
    y_train, y_test = d["y_train"], d["y_test"]
    B = x_train.shape[0]
    reg = TabPFNRegressor(random_state=42)
    output = []
    for i in range(B):
        reg.fit(x_train[i], y_train[i])
        f_lower, f_mu, f_upper = reg.predict(
            x_test[i],
            output_type="quantiles",
            quantiles=[0.025, 0.5, 0.975],
        )
        rmse = np.sqrt(np.square(f_mu - y_test[i]).mean())
        mae = np.abs(f_mu - y_test[i])
        cvg = (y_test[i] >= f_lower) & (y_test[i] <= f_upper)
        output += [{"rmse": rmse, "mae": mae, "cvg": cvg}]
    np.save(path, output)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", help="Path to <xydata>.npy file.")
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args.path)
