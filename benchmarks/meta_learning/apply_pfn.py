#!/usr/bin/env -S PYENV_VERSION=torch python3
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tqdm import tqdm

# NOTE: sometimes TabPFN.predict_proba values don't sum to one exactly and this
# throws constant warnings...ignore these
warnings.filterwarnings(
    "ignore",
    message="The y_pred values do not sum to one.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Found unknown categories in columns.*",
    category=UserWarning,
)


def main(path, use_classifier):
    method = classify if use_classifier else regress
    method(path)


def regress(path):
    model = TabPFNRegressor(random_state=42)
    output = []
    batches = np.load(path, allow_pickle=True)
    pbar = tqdm(batches, unit=" batches", leave=False, dynamic_ncols=True)
    for b in pbar:
        x_train, x_test = b["x_train"], b["x_test"]
        y_train, y_test = b["y_train"], b["y_test"]
        mask_train, mask_test = b["mask_train"], b["mask_test"]
        y_dim = y_train.shape[-1]
        for i in range(x_train.shape[0]):  # iterate over batch tasks
            mses, maes, cvgs = 0, 0, 0
            for j in range(y_dim):
                y_train_ij = y_train[i][mask_train[i]][:, j]
                y_test_ij = y_test[i][mask_test[i]][:, j]
                model.fit(x_train[i][mask_train[i]], y_train_ij)
                f_lower, f_mu, f_upper = model.predict(
                    x_test[i][mask_test[i]],
                    output_type="quantiles",
                    quantiles=[0.025, 0.5, 0.975],
                )
                mses += np.square(f_mu - y_test_ij).mean()
                maes += np.abs(f_mu - y_test_ij).mean()
                cvgs += ((y_test_ij >= f_lower) & (y_test_ij <= f_upper)).mean()
            output += [
                {
                    "rmse": np.sqrt(np.mean(mses)),
                    "mae": np.mean(maes),
                    "cvg": np.mean(cvgs),
                }
            ]
    with open(Path(path).parent / "pfn_results.json", "w") as f:
        json.dump(pd.DataFrame(output).mean().to_dict(), f)


def classify(path):
    model = TabPFNClassifier(random_state=42)
    output = []
    batches = np.load(path, allow_pickle=True)
    pbar = tqdm(batches, unit=" batches", leave=False, dynamic_ncols=True)
    for b in pbar:
        x_train, x_test = b["x_train"], b["x_test"]
        y_train, y_test = b["y_train"], b["y_test"]
        mask_train, mask_test = b["mask_train"], b["mask_test"]
        num_classes = y_train[0][0].shape[-1]
        # NOTE: this assumes that classes range from 0 to num_classes-1
        classes = list(range(num_classes))
        for i in range(x_train.shape[0]):  # iterate over batch tasks
            y_train_i = y_train[i][mask_train[i]]
            y_test_i = y_test[i][mask_test[i]]
            if num_classes > 1:
                # convert one-hot back to 1D array of class labels
                y_train_i = np.argmax(y_train_i, axis=-1).flatten()
                y_test_i = np.argmax(y_test_i, axis=-1).flatten()
            model.fit(x_train[i][mask_train[i]], y_train_i)
            proba = model.predict_proba(x_test[i][mask_test[i]])
            # this is necessary so that if the context set only
            # contains a subset of labels, TabPFN will only output
            # probabilities for the seen labels
            full_proba = np.zeros((proba.shape[0], num_classes))
            for i, cls in enumerate(model.classes_):
                full_proba[:, cls] = proba[:, i]
            nll = log_loss(y_test_i, full_proba, labels=classes)
            output += [{"nll": nll}]
    with open(Path(path).parent / "pfn_results.json", "w") as f:
        json.dump(pd.DataFrame(output).mean().to_dict(), f)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", help="Path to <xydata>.npy file.")
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Use a classifier instead of a regressor.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args.path, args.classify)
