#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def main(args):
    path = Path(args.csv)
    tx, grid = prepare_data(path)
    plot(grid, path.with_suffix(".png"))
    m = load_ckpt(args.ckpt)
    m = finetune(m, grid)
    post_pred = infer(m, grid)
    with open("diagonal_posterior_predictive.pkl", "wb") as f:
        pickle.dump(post_pred, f)
    plot(grid, path.with_stem(path.stem + "_with_posterior_mean").with_suffix(".png"))


def prepare_data(csv: Path) -> tuple[ColumnTransformer, np.ndarray]:
    df = pd.read_csv(csv)
    return preprocess(df)


def preprocess(df: pd.DataFrame) -> tuple[ColumnTransformer, np.ndarray]:
    """Normalize `Lat`/`Lon` to `[0, 1]` and standardize `Temp`."""
    df = df.sort_values(["Lat", "Lon"], ascending=[False, True])
    tx = ColumnTransformer(
        [
            ("normalizer", MinMaxScaler((-2, 2)), ["Lat", "Lon"]),
            ("standardizer", StandardScaler(), ["Temp"]),
        ],
        verbose_feature_names_out=False,
    )
    grid = tx.fit_transform(df).reshape(300, 500, 3)
    return tx, grid


def load_ckpt() -> tuple[TrainState, DictConfig]:
    # TODO(danj): implement
    pass


def finetune(m: TrainState, grid: np.ndarray) -> TrainState:
    # TODO(danj): implement
    pass


def infer(m: TrainState, grid: np.ndarray) -> np.ndarray:
    # TODO(danj): implement
    pass


def plot(grid: np.ndarray, path: Path):
    plt.imshow(grid[..., -1], cmap="inferno", interpolation="none")
    plt.savefig(path, dpi=600)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--csv",
        default="data/sim.csv",
        help="Path to benchmark csv.",
    )
    parser.add_argument(
        "-k",
        "--ckpt",
        default="results/gp/2d/rbf/42/TNPKR.pkl",
        help="Path to model checkpoint.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args.csv)
