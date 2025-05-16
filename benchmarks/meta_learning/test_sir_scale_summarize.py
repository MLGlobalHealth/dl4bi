#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import pandas as pd

# Example: python test_sir_scale_summarize.py "BSA-TNP - SIR"


def main():
    dfs = []
    project = sys.argv[1]
    results_path = Path(f"results/{project}")
    for path in results_path.rglob("*.json"):
        if path.stem.startswith("_"):  # checkpoint json files
            continue
        model_name, img_size = path.stem.split("_")
        with open(path) as f:
            d = json.load(f)
        num_obs = len(d["NLL"])
        d["model"] = [model_name] * num_obs
        d["img_size"] = [img_size] * num_obs
        df = pd.DataFrame(d)
        dfs += [df]
    df = pd.concat(dfs)
    df["ms_elapsed"] = df["s_elapsed"] * 1000
    process(df, "NLL")
    process(df, "ms_elapsed", precision=1)


def process(df, stat, precision=3):
    df = df[["model", "img_size", stat]]
    df = df.groupby(["model", "img_size"]).agg(["mean", "sem"])
    df.columns = ["mean", "stderr"]
    df = df.reset_index()
    df[stat] = df.apply(
        lambda r: f"${r['mean']:0.{precision}f}\\pm{r['stderr']:.{precision}f}$", axis=1
    )
    df = df[["model", "img_size", stat]]
    df = df.pivot(index="model", columns=["img_size"])
    df.columns = df.columns.droplevel(0)
    df = df[["128x128", "256x256", "1024x1024"]]
    df = df.fillna("OOM")
    df = df.reset_index()
    df_tex = df.to_latex(
        index=False,
        caption=f"SIR {stat}.",
        label=f"sir_scale_{stat}",
        column_format="l" + "r" * (len(df.columns) - 1),
    )
    print(df_tex)


if __name__ == "__main__":
    main()
