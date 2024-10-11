#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    dfs = []
    for path in Path("results/gp").rglob("*.npy"):
        record = list(path.parts)
        record = record[record.index("gp") + 1 : -1] + [path.stem]
        regret = np.load(path)[:, -1]  # final regret
        df = pd.DataFrame([record] * len(regret))
        df["regret"] = regret
        dfs += [df]
    df = pd.concat(dfs)
    df.columns = ["gp", "kernel", "seed", "model", "regret"]
    df = df.groupby(["gp", "kernel", "model"]).agg({"regret": ["mean", "sem"]})
    df.columns = ["mean", "stderr"]
    df = df.apply(lambda r: f"${r['mean']:0.3f}\\pm{r['stderr']:.3f}$", axis=1)
    df = df.reset_index()
    colnames = list(df.columns)
    colnames[-1] = "regret"
    df.columns = colnames
    df = df.pivot(index="model", columns=["gp", "kernel"], values="regret")
    df = df.reset_index()
    df_tex = df.to_latex(
        index=False,
        caption="1D Bayesian Optimization",
        label="bo_1d",
        column_format="l" + "r" * (len(df.columns) - 1),
    )
    print(df_tex)


if __name__ == "__main__":
    main()
