#!/usr/bin/env -S PYENV_VERSION=torch python3
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import wandb
from tabpfn import TabPFNRegressor


def main(project, seed):
    model = TabPFNRegressor(random_state=42)
    path = Path(f"results/{project}/{seed}/MCMC_sample.pkl")
    with open(path, "rb") as fp:
        sample = pickle.load(fp)
    stack = lambda *args: np.concat(args, axis=-1)
    x_train = stack(sample["s_ctx"], sample["x_ctx"])
    x_test = stack(sample["s_test"], sample["x_test"])
    y_train = sample["f_ctx"].flatten()
    y_test = sample["f_test"].flatten()
    model.fit(x_train, y_train)
    y_lower, y_mu, y_upper = model.predict(
        x_test,
        output_type="quantiles",
        quantiles=[0.025, 0.5, 0.975],
    )
    metrics = {
        "RMSE": np.sqrt(np.square(y_test - y_mu).mean()),
        "MAE": np.abs(y_test - y_mu).mean(),
        "CVG": ((y_test >= y_lower) & (y_test <= y_upper)).mean(),
    }
    wandb.init(
        config={"seed": seed},
        mode="online",
        name="TabPFN - Infer",
        project=args.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    wandb.log({f"Infer {m}": v.item() for m, v in metrics.items()})


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("seed", type=int, default=46, help="The seed.")
    parser.add_argument(
        "-p",
        "--project",
        default="NeurIPS BSA-TNP - Generic Spatial",
        help="The project with the MCMC_sample.pkl files.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args.project, args.seed)
