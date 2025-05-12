#!/usr/bin/env -S PYENV_VERSION=torch python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor
from tqdm import tqdm


def main(path):
    reg = TabPFNRegressor(random_state=42)
    output = []
    batches = np.load(path, allow_pickle=True)
    pbar = tqdm(batches[:10], unit=" batches", leave=False, dynamic_ncols=True)
    for b in pbar:
        x_train, x_test = b["x_train"], b["x_test"]
        y_train, y_test = b["y_train"], b["y_test"]
        mask_train, mask_test = b["mask_train"], b["mask_test"]
        for i in range(x_train.shape[0]):  # iterate over batch tasks
            y_train_i = y_train[i][mask_train[i]].flatten()
            y_test_i = y_test[i][mask_test[i]].flatten()
            reg.fit(x_train[i][mask_train[i]], y_train_i)
            f_lower, f_mu, f_upper = reg.predict(
                x_test[i][mask_test[i]],
                output_type="quantiles",
                quantiles=[0.025, 0.5, 0.975],
            )
            rmse = np.sqrt(np.square(f_mu - y_test_i).mean())
            mae = np.abs(f_mu - y_test_i).mean()
            cvg = ((y_test_i >= f_lower) & (y_test_i <= f_upper)).mean()
            output += [{"rmse": rmse, "mae": mae, "cvg": cvg}]
    with open(Path(path).parent / "pfn_results.json", "w") as f:
        json.dump(pd.DataFrame(output).mean().to_dict(), f)
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
