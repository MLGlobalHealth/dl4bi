#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis.py
-----------
Generates main-paper and appendix tables/plots for scalability experiments.

Features:
- Auto-load CSVs from results/scalability_log_priors_ls_{ls}/aggregated_results.csv
- Best-model (and top-2) per (ls, grid_size, metric) tables (CSV + LaTeX)
- Main plots: line/grouped-bar per metric vs grid_size; scatter trade-off
- Appendix: full per-lengthscale tables; plus lines/scatter with all models
- Clean directory structure for outputs
- Name shortcut mapping for cleaner figure legends & tables
- Uses jax.numpy as jnp for reductions/rankings

Run:
    python analysis.py --base_dir results --out_dir outputs \
        --lengthscales 10 30 50 \
        --grid_sizes 256 576 1024 2304 4096 \
        --style line \
        --highlight_models ProposedModel Baseline_GP

Dependencies: pandas, numpy, jax, seaborn, matplotlib
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ----------------------------
# Configuration / Shortcuts
# ----------------------------


DEFAULT_PLOT_METRICS = [
    "MSE(y_hat_gp, y_hat)",
    "ls wasserstein distance",
    "ESS ls",
    "ESS beta",
    "unobs MSE(y, y_hat)",
    "obs MSE(y, y_hat)",
    "unobs MSE(y, y_hat)",
]

DEFAULT_MAIN_TABLE_METRICS = [
    "MSE(y_hat_gp, y_hat)",
    "ls wasserstein distance",
    "ESS ls per second",
    "obs MSE(y, y_hat)",
    "unobs MSE(y, y_hat)",
    "infer_time",
]

DEFAULT_APPENDIX_TABLE_METRICS = [
    "Test Norm MSE",
    "infer_time",
    "total_time",
    "MSE(y, y_hat)",
    "ESS ls",
    "ESS beta",
    "obs MSE(y, y_hat)",
    "unobs MSE(y, y_hat)",
    "MSE(y_hat_gp, y_hat)",
    "ls wasserstein distance",
    "beta wasserstein distance",
    "ls",
    "ESS ls per second",
]
shortcut_names: Dict[str, str] = {
    "Baseline_GP": "GP",
    "DeepRV + gMLP kAttn": "DeepRV - gMLP kAttn",
    "DeepRV + gMLP adamw": "DeepRV - gMLP",
    "DeepRV + gMLP": "DeepRV - gMLP",
    "DeepRV + MLP": "DeepRV - MLP",
    "Inducing Points": "Inducing Pts small",
    "Inducing Points Large": "Inducing Pts",
    "DeepRV + trans kAttn": "DeepRV - trans kAttn",
}

metric_display_names: Dict[str, str] = {
    "MSE(y_hat_gp, y_hat)": r"MSE($\mathbf{\hat{y}}_{gp}, \mathbf{\hat{y}}$)",
    "ls wasserstein distance": r"Wass($\hat{\ell}_{gp}, \hat{\ell}$)",
    "beta wasserstein distance": r"Wass($\hat{\beta}_{gp}, \hat{\beta}$)",
    "infer_time": "Inference Time (s)",
    "ESS ls": r"ESS ($\ell$)",
    "ESS beta": r"ESS ($\beta$)",
    "ESS ls per second": r"$\text{ESS } (\ell)/{\text{sec}}$",
    "ESS beta per second": r"$\text{ESS } (\beta)/{\text{sec}}$",
    "MSE(y, y_hat)": r"MSE($\mathbf{y}, \mathbf{\hat{y}}$)",
    "obs MSE(y, y_hat)": r"Observed MSE($\mathbf{y}, \mathbf{\hat{y}}$)",
    "unobs MSE(y, y_hat)": r"Unobserved MSE($\mathbf{y}, \mathbf{\hat{y}}$)",
    "grid_size": "Grid",
    "train_time": "Train Time (s)",
    "Test Norm MSE": "Test Loss",
}

hyperparam_display_names: Dict[str, str] = {
    "beta": r"$\beta$",
    "alpha": r"$\alpha$",
    "nu": r"$\nu$",
    "var": r"$\sigma$",
    "ls": r"$\ell$",
}


# ----------------------------
# Utilities
# ----------------------------


def display_name(model_name: str) -> str:
    return shortcut_names.get(model_name, model_name)


def display_metric(metric: str) -> str:
    return metric_display_names.get(metric, metric)


def display_hyperparam(var: str) -> str:
    return hyperparam_display_names.get(var, var)


def is_lower_better(metric: str) -> bool:
    lower_better_keywords = ["mse", "distance", "time", "flops", "wasserstein"]
    metric_lower = metric.strip().lower()
    if "ess" in metric_lower:
        return False
    return any(k in metric_lower for k in lower_better_keywords)


def topk_per_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str,
    k: int = 1,
    lower_is_better: Optional[bool] = None,
) -> pd.DataFrame:
    if lower_is_better is None:
        lower_is_better = is_lower_better(value_col)

    work = df.dropna(subset=[value_col]).copy()

    def _rank_block(block: pd.DataFrame) -> pd.DataFrame:
        vals = block[value_col].to_numpy()
        order = jnp.argsort(vals)
        if not lower_is_better:
            order = order[::-1]
        order = np.array(order)
        return block.iloc[order[: min(k, len(order))]]

    ranked = work.groupby(list(group_cols), group_keys=False).apply(_rank_block)
    return ranked.reset_index(drop=True)


def filter_models(df: pd.DataFrame, models: Optional[Sequence[str]]) -> pd.DataFrame:
    if models is None:
        return df
    return df[df["model_name"].isin(models)].copy()


def _apply_theme():
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.6)


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Data Loading
# ----------------------------


def load_all_results(
    base_dir: Path, dir_prefix: str, lengthscales: Sequence[int]
) -> pd.DataFrame:
    dfs = []
    for ls in lengthscales:
        csv_path = base_dir / f"{dir_prefix}_ls_{ls}" / "aggregated_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing results CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        df["ls"] = ls
        dfs.append(df)
    out = pd.concat(dfs, axis=0, ignore_index=True)
    return out


def ensure_grid_sizes(
    df: pd.DataFrame, expected_grid_sizes: Optional[Sequence[int]] = None
) -> pd.DataFrame:
    if "grid_size" not in df.columns:
        raise ValueError("Expected column 'grid_size' not found in the CSVs.")
    if expected_grid_sizes:
        df = df[df["grid_size"].isin(expected_grid_sizes)].copy()
    return df


# ----------------------------
# Tables
# ----------------------------


def make_best_model_table(
    df: pd.DataFrame,
    metrics: Sequence[str],
    topk: int = 1,
    models_subset: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - detailed table: per (ls, grid_size)
      - aggregated table (per ls): mean over grid_size
      - aggregated table (per grid_size): mean over ls
    """
    work = df.copy()
    if models_subset is not None:
        work = work[work["model_name"].isin(models_subset)]

    rows_detailed = []
    rows_agg_ls = []
    rows_agg_grid = []

    for metric in metrics[::-1]:
        if metric not in work.columns:
            continue

        lower_better = is_lower_better(metric)

        filt = work
        if "ess" in metric.lower():
            filt = work[~work["model_name"].str.contains("ADVI", case=False)]

        # -------- detailed version (per ls, grid_size) --------
        ranked = topk_per_group(
            filt,
            group_cols=["ls", "grid_size"],
            value_col=metric,
            k=topk,
            lower_is_better=lower_better,
        )

        for (ls, grid_size), block in ranked.groupby(["ls", "grid_size"]):
            block_sorted = block.sort_values(metric, ascending=lower_better)
            entry = {
                "ls": ls,
                "grid_size": grid_size,
                "metric": display_metric(metric),
            }
            for i, (_, r) in enumerate(block_sorted.head(topk).iterrows(), start=1):
                entry[f"top{i}_model"] = display_name(r["model_name"])
            rows_detailed.append(entry)

        # -------- aggregated version (mean over grid_size, per ls) --------
        agg_ls = (
            filt.groupby(["ls", "model_name"], as_index=False)[metric]
            .mean()
            .rename(columns={metric: "mean_metric"})
        )

        ranked_ls = topk_per_group(
            agg_ls,
            group_cols=["ls"],
            value_col="mean_metric",
            k=topk,
            lower_is_better=lower_better,
        )

        for ls, block in ranked_ls.groupby("ls"):
            block_sorted = block.sort_values("mean_metric", ascending=lower_better)
            entry = {
                "ls": ls,
                "metric": display_metric(metric),
            }
            for i, (_, r) in enumerate(block_sorted.head(topk).iterrows(), start=1):
                entry[f"top{i}_model"] = display_name(r["model_name"])
            rows_agg_ls.append(entry)

        # -------- aggregated version (mean over ls, per grid_size) --------
        agg_grid = (
            filt.groupby(["grid_size", "model_name"], as_index=False)[metric]
            .mean()
            .rename(columns={metric: "mean_metric"})
        )

        ranked_grid = topk_per_group(
            agg_grid,
            group_cols=["grid_size"],
            value_col="mean_metric",
            k=topk,
            lower_is_better=lower_better,
        )

        for grid_size, block in ranked_grid.groupby("grid_size"):
            block_sorted = block.sort_values("mean_metric", ascending=lower_better)
            entry = {
                "grid_size": grid_size,
                "metric": display_metric(metric),
            }
            for i, (_, r) in enumerate(block_sorted.head(topk).iterrows(), start=1):
                entry[f"top{i}_model"] = display_name(r["model_name"])
            rows_agg_grid.append(entry)

    out_detailed = (
        pd.DataFrame(rows_detailed)
        .sort_values(["metric", "ls", "grid_size"])
        .reset_index(drop=True)
    )

    out_agg_ls = (
        pd.DataFrame(rows_agg_ls).sort_values(["metric", "ls"]).reset_index(drop=True)
    )

    out_agg_grid = (
        pd.DataFrame(rows_agg_grid)
        .sort_values(["metric", "grid_size"])
        .reset_index(drop=True)
    )

    return out_detailed, out_agg_ls, out_agg_grid


def make_aggregated_model_table(
    df: pd.DataFrame,
    metrics: Sequence[str],
    models_subset: Optional[Sequence[str]] = None,
    groupby: str = "ls",  # or "grid_size", or "both"
) -> pd.DataFrame:
    """
    Aggregates metrics over ls/grid_size and reports mean ± std for each model.
    NaNs replaced with '-'.
    Best model (lowest mean) and those within 1 std of it are bolded.
    Adds midline between metrics, and writes metric name only in middle row.
    """
    work = df.copy()
    if models_subset is not None:
        work = work[work["model_name"].isin(models_subset)]
    tables = []
    for metric in metrics:
        if metric not in work.columns:
            continue
        group_cols = []
        if groupby in ("ls", "both"):
            group_cols.append("ls")
        if groupby in ("grid_size", "both"):
            group_cols.append("grid_size")
        agg = (
            work.groupby(group_cols + ["model_name"], as_index=False)[metric]
            .agg(mean="mean", se=lambda x: x.std(ddof=1) / (len(x) ** 0.5))
            .reset_index()
        )
        rounding = 3
        if "infer_time" in metric:
            rounding = 0
        if "wass" in metric.lower() or "ESS" in metric:
            rounding = 2
        agg["formatted"] = agg.apply(
            lambda r: f"{jnp.round(r['mean'], rounding):.{rounding}f} ±"
            f" {jnp.round(r['se'], 2):.2f}"
            if not pd.isna(r["mean"])
            else "-",
            axis=1,
        )
        pivoted = agg.pivot(
            index=group_cols, columns="model_name", values="formatted"
        ).reset_index()
        pivoted.rename(columns=display_name, inplace=True)
        n_rows = len(pivoted)
        mid_idx = n_rows // 2
        pivoted.insert(0, "metric", [""] * n_rows)
        pivoted.loc[mid_idx, "metric"] = display_metric(metric)
        midline = {c: r"\midrule" for c in pivoted.columns}
        pivoted = pd.concat([pivoted, pd.DataFrame([midline])], ignore_index=True)
        tables.append(pivoted)
    out = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return out


def export_table(df: pd.DataFrame, csv_path: Path, latex_path: Optional[Path] = None):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if latex_path is not None:
        with open(latex_path, "w") as f:
            f.write(df.to_latex(index=False, escape=False))


def per_lengthscale_full_table(
    df: pd.DataFrame, ls: int, metrics: list[str]
) -> pd.DataFrame:
    """
    Build per-lengthscale full table restricted to given metrics.
    Keeps 'grid_size' and 'model_name' as identifiers.
    """
    tab = df[df["ls"] == ls].copy()
    id_cols = ["grid_size", "model_name"]

    # Only keep requested metrics + id_cols (if present in df)
    keep_cols = [c for c in id_cols + metrics if c in tab.columns]

    tab = tab[keep_cols]
    tab = tab.sort_values(["grid_size", "model_name"]).reset_index(drop=True)
    return tab


def make_ablation_latex_table(
    df: pd.DataFrame, metric: str, rounding: int = 3
) -> pd.DataFrame:
    if metric == "Test Norm MSE":
        rounding = 5
    df = df[(df.model_name != "DeepRV + gMLP kAttn")].reset_index(drop=True)
    if "gp" in metric or "wass" in metric or "Test" in metric:
        df = df[(df.model_name != "GP")].reset_index(drop=True)

    agg = (
        df.groupby(["model_name", "kernel"], as_index=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["SE"] = agg.apply(
        lambda r: r["std"] / jnp.sqrt(float(r["count"])) if r["count"] > 0 else jnp.nan,
        axis=1,
    )

    higher_better = "ESS" in metric

    # create formatted string and bold best
    def fmt_val(row, best_per_kernel):
        if pd.isna(row["mean"]):
            return "-"
        val_str = f"{row['mean']:.{rounding}f} ± {row['SE']:.{max(rounding, 2)}f}"
        if row["mean"] in best_per_kernel[row["kernel"]]:
            val_str = f"\\mathbf{{{val_str}}}"
        return val_str

    # determine best per kernel
    best_per_kernel = {}
    for k in df["kernel"].unique():
        subset = agg[agg["kernel"] == k]
        if higher_better:
            best_val = subset["mean"].max()
        else:
            best_val = subset["mean"].min()
        best_per_kernel[k] = [best_val]

    agg["formatted"] = agg.apply(lambda r: fmt_val(r, best_per_kernel), axis=1)

    final_table = agg.pivot(
        index="model_name", columns="kernel", values="formatted"
    ).reset_index()
    final_table.rename(columns={"model_name": "Model"}, inplace=True)
    final_table["Model"] = final_table["Model"].map(display_name)
    final_table.columns = [
        display_metric(c) if c != "Model" else "Model" for c in final_table.columns
    ]

    return final_table


def export_ablation_table(
    df: pd.DataFrame, metric: str, save_dir: Path, filename: Optional[str] = None
):
    """
    Aggregates ablation results and exports CSV + LaTeX using pipeline's export_table.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    table = make_ablation_latex_table(df, metric)
    fname = filename or f"ablation_{metric.replace(' ', '_')}"
    csv_path = save_dir / f"{fname}.csv"
    latex_path = save_dir / f"{fname}.tex"
    export_table(table, csv_path, latex_path)


def _make_table_from_metrics(
    df: pd.DataFrame,
    metrics: list[tuple[str, bool, int, str]],
) -> pd.DataFrame:
    """
    Build a single LaTeX-ready table from a list of metrics.
    Each metric tuple: (metric_name_in_df, higher_better, rounding, display_name)

    - Filters:
        * Drop "DeepRV + gMLP kAttn"
        * Drop "GP" when metric contains "gp", "wass", or "test" (case-insensitive)
    - Aggregation: averages across kernels/runs (groupby model_name)
    - Formatting: mean ± SE, bold best per column.
    """
    table = pd.DataFrame()

    for metric, higher_better, rounding in metrics:
        sub = df.copy()
        sub = sub[(sub.model_name != "DeepRV + gMLP kAttn")].reset_index(drop=True)
        if (
            ("gp" in metric.lower())
            or ("wass" in metric.lower())
            or ("test" in metric.lower())
        ):
            sub = sub[(sub.model_name != "GP")].reset_index(drop=True)

        # aggregate across kernels/runs
        agg = (
            sub.groupby(["model_name"], as_index=False)[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg["SE"] = agg.apply(
            lambda r: r["std"] / jnp.sqrt(float(r["count"]))
            if r["count"] > 0
            else jnp.nan,
            axis=1,
        )

        # choose best and format
        best_val = agg["mean"].max() if higher_better else agg["mean"].min()
        tol = 1e-12

        def fmt(row):
            if pd.isna(row["mean"]):
                return "-"
            s = f"{row['mean']:.{rounding}f} ± {row['SE']:.{max(rounding, 2)}f}"
            return f"\\mathbf{{{s}}}" if abs(row["mean"] - best_val) <= tol else s

        nice_name = metric_display_names.get(metric, metric)
        col = pd.DataFrame(
            {
                "model_name": agg["model_name"],
                nice_name: agg.apply(fmt, axis=1),
            }
        )
        table = col if table.empty else table.merge(col, on="model_name", how="outer")

    table.rename(columns={"model_name": "Model"}, inplace=True)
    table["Model"] = table["Model"].map(display_name)
    # If you want to map headers through display_metric, uncomment:
    table.columns = [
        display_metric(c) if c != "Model" else "Model" for c in table.columns
    ]
    return table


def run_ablation():
    df = pd.read_csv("results/ablation_test/res.csv")
    for col in df.columns:
        if "ess" in col.lower() or col.lower() == "infer_time":
            df[col] = df[col] / df["num_chains"]
    for col in df.columns:
        if "ess" in col.lower():
            df[col + " per second"] = df[col] / df["infer_time"]

    save_dir = Path("outputs/ablation_tables")
    for metric in [
        "Test Norm MSE",
        "infer_time",
        "MSE(y, y_hat)",
        "obs MSE(y, y_hat)",
        "unobs MSE(y, y_hat)",
        "MSE(y_hat_gp, y_hat)",
        "ls wasserstein distance",
        "ESS ls per second",
    ]:
        export_ablation_table(df, metric, save_dir)

    # ---- MAIN TABLES (outside the loop) ----
    # Table A: Performance / Fidelity
    metrics_A = [
        ("MSE(y_hat_gp, y_hat)", False, 3),
        ("Test Norm MSE", False, 5),
        ("ls wasserstein distance", False, 3),
    ]
    table_A = _make_table_from_metrics(df, metrics_A)
    export_table(
        table_A,
        save_dir / "ablation_main_table_A.csv",
        save_dir / "ablation_main_table_A.tex",
    )

    # Table B: Efficiency (incl. train_time)
    metrics_B = [
        ("ESS ls per second", True, 3),
        ("infer_time", False, 2),
        ("train_time", False, 1),
    ]
    table_B = _make_table_from_metrics(df, metrics_B)
    export_table(
        table_B,
        save_dir / "ablation_main_table_B.csv",
        save_dir / "ablation_main_table_B.tex",
    )


# ----------------------------
# Plotting
# ----------------------------


@dataclass
class PlotStyle:
    kind: str = "line"  # "line" or "bar"


def plot_metric_vs_grid_size(
    df: pd.DataFrame,
    metric: str,
    save_path: Path,
    models: Optional[Sequence[str]] = None,
    style: PlotStyle = PlotStyle("line"),
):
    _apply_theme()
    work = filter_models(df, models)
    work = work[~work[metric].isna()]
    if metric not in work.columns:
        print(f"[plot_metric_vs_grid_size] Metric '{metric}' not found; skipping.")
        return

    work = work.copy()
    work["model_display"] = work["model_name"].map(display_name)
    work = work.sort_values(["ls", "model_display", "grid_size"])

    g = None
    if style.kind == "bar":
        g = sns.catplot(
            data=work,
            x="grid_size",
            y=metric,
            hue="model_display",
            col="ls",
            kind="bar",
            sharey=False,
            height=4,
            aspect=1.1,
        )
    else:
        g = sns.relplot(
            data=work,
            x="grid_size",
            y=metric,
            hue="model_display",
            col="ls",
            kind="line",
            marker="o",
            facet_kws={"sharey": False},
            height=4,
            aspect=1.1,
        )

    g.set_axis_labels("Grid Size", display_metric(metric))
    g.set_titles("ls = {col_name}")
    g._legend.set_title("")  # remove title
    for t in g._legend.texts:
        t.set_fontsize(8)
    g._legend.set_bbox_to_anchor((0.95, 0.85))
    g._legend.set_loc("upper right")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(save_path, dpi=300)
    plt.close(g.fig)


def plot_lines_or_bars_for_metrics(
    df: pd.DataFrame,
    metrics: Sequence[str],
    out_dir: Path,
    models: Optional[Sequence[str]],
    style: PlotStyle,
):
    for metric in metrics:
        if metric == "infer_time":
            continue
        fname = (
            metric.replace(" ", "_")
            .replace("/", "_")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
        )
        save_path = out_dir / f"{style.kind}_{fname}.png"
        plot_metric_vs_grid_size(df, metric, save_path, models=models, style=style)


def plot_aggregated_bar(
    df: pd.DataFrame,
    metric: str,
    save_path: Path,
    models: Optional[Sequence[str]] = None,
    base_model_name: Optional[str] = "Baseline_GP",
    quantiles: tuple = (0.1, 0.9),
):
    _apply_theme()
    work = df.copy()
    unique_models = work["model_name"].map(display_name).unique()
    colors = sns.color_palette("tab10", len(unique_models))
    palette = dict(zip(unique_models, colors))
    if base_model_name is not None:
        work = work[work["model_name"] != base_model_name]
    if models is not None:
        work = work[work["model_name"].isin(models)]
    if metric not in work.columns:
        print(f"[plot_aggregated_bar] Metric {metric} not found; skipping.")
        return
    # --- compute mean and quantiles ---
    q_low, q_high = quantiles
    agg_stats = (
        work.groupby("model_name")[metric]
        .agg(
            mean="mean",
            q_low=lambda x: x.quantile(q_low),
            q_high=lambda x: x.quantile(q_high),
        )
        .reset_index()
    )
    agg_stats = agg_stats[~pd.isna(agg_stats["mean"])].reset_index(drop=True)
    agg_stats["model_display"] = agg_stats["model_name"].map(display_name)
    agg_stats = agg_stats.sort_values("mean")
    # --- raw values for plotting ---
    y_mean = agg_stats["mean"].values
    y_low = agg_stats["q_low"].values
    y_high = agg_stats["q_high"].values
    # --- bar plot with error bars ---
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.bar(
        x=agg_stats["model_display"],
        height=y_mean,
        yerr=[y_mean - y_low, jnp.maximum(y_high - y_mean, 1e-12)],
        capsize=4,
        color=[palette[m] for m in agg_stats["model_display"]],
    )
    ax.set_yscale("log", base=2)
    ax.set_xlabel("")
    ax.set_title(
        f"{r'$\text{Mean}$'} {display_metric(metric).replace('Wass', 'Wasserstein')}"
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_mean_infer_time(
    df: pd.DataFrame, save_path: Path, models: Optional[Sequence[str]] = None
):
    _apply_theme()
    work = filter_models(df, models)
    if "infer_time" not in work.columns:
        return
    agg = work.groupby(["model_name", "grid_size"], as_index=False)["infer_time"].mean()
    agg["model_display"] = agg["model_name"].map(display_name)
    agg["log_infer_time"] = jnp.log(jnp.array(agg["infer_time"].values))
    g = sns.relplot(
        data=agg,
        x="grid_size",
        y="log_infer_time",
        hue="model_display",
        kind="line",
        marker="o",
    )
    g.set_axis_labels("Grid Size", "Mean Inference Time (s)")
    for ax in g.axes.flat:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"$e^{{{int(round(y))}}}$")
        )
    g._legend.set_title("")
    for t in g._legend.texts:
        t.set_fontsize(8)
    g._legend.set_bbox_to_anchor((0.1, 0.9))
    g._legend.set_loc("upper left")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(save_path, dpi=300)
    plt.close(g.fig)


def plot_posterior_predictive_comparisons_kde(
    samples: list[dict],
    model_names: list[str],
    var_names: list[str],
    save_prefix: Path,
):
    n_vars = len(var_names)
    fig, axes = plt.subplots(1, n_vars, figsize=(n_vars * 1.7, 2.0), sharey=False)
    if n_vars == 1:
        axes = [axes]
    all_handles, all_labels = None, None
    for i, var_name in enumerate(var_names):
        ax = axes[i]
        min_val, max_val = np.inf, -np.inf
        for model_name, model_dict in zip(model_names, samples):
            model_samples = model_dict.get(str(var_name), None)
            if model_samples is not None:
                if var_name == "a":
                    model_samples = model_samples[model_samples <= 2]
                min_val = min(min_val, model_samples.min())
                max_val = max(max_val, model_samples.max())
                model_n = model_name.split(" +")[0]
                sns.kdeplot(
                    model_samples, label=model_n, linewidth=1.2, alpha=0.7, ax=ax
                )
        ax.set_xlabel(display_hyperparam(var_name), fontsize=9)
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)
        if i == 0:
            all_handles, all_labels = ax.get_legend_handles_labels()
    if all_handles:
        fig.legend(
            all_handles,
            all_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),  # closer to plots
            ncol=len(all_labels),
            fontsize=8,
            frameon=False,
        )
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave just enough space for legend
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{save_prefix}_posterior_kde.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_speed_quadrant(
    df: pd.DataFrame,
    save_path: Path,
    accuracy_metrics: Optional[Sequence[str]] = [
        "MSE(y_hat_gp, y_hat)",
        "ls wasserstein distance",
    ],
    models: Optional[Sequence[str]] = None,
):
    _apply_theme()
    work = filter_models(df, models)
    if any("gp" in m or "wass" in m for m in accuracy_metrics):
        work = work[work.model_name != "Baseline_GP"].reset_index(drop=True)
    required_cols = ["infer_time"] + list(accuracy_metrics)
    if not all(col in work.columns for col in required_cols):
        print("[plot_accuracy_vs_speed_quadrant] Required columns not found.")
        return
    agg = work.groupby("model_name", as_index=False)[required_cols].mean()
    agg["model_display"] = agg["model_name"].map(display_name)
    x = agg["infer_time"].values
    x_scaled = (x - x.mean()) / x.std()
    agg["infer_time_scaled"] = x_scaled
    acc_norms = []
    for m in accuracy_metrics:
        vals = agg[m].values
        vals_scaled = (vals - vals.mean()) / vals.std()
        acc_norms.append(vals_scaled)
    y_combined = np.sum(acc_norms, axis=0)
    y_scaled = (y_combined - y_combined.mean()) / y_combined.std()
    agg["accuracy_scaled"] = y_scaled
    x_med, y_med = np.median(x_scaled), np.median(y_scaled)
    points = np.column_stack([x_scaled, y_scaled])
    is_efficient = np.ones(points.shape[0], dtype=bool)
    for i, (xi, yi) in enumerate(points):
        if np.any(
            (x_scaled <= xi) & (y_scaled <= yi) & ((x_scaled < xi) | (y_scaled < yi))
        ):
            is_efficient[i] = False
    pareto_points = points[is_efficient]
    pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]
    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    sns.scatterplot(
        data=agg,
        x="infer_time_scaled",
        y="accuracy_scaled",
        hue="model_display",
        style="model_display",
        s=60,
        ax=ax,
    )
    ax.axvline(x_med, ls="--", lw=1, c="grey")
    ax.axhline(y_med, ls="--", lw=1, c="grey")
    ax.set_xlabel(f"Speed (normalized {display_metric('infer_time')})")
    acc_title = " + ".join(display_metric(m) for m in accuracy_metrics)
    ax.set_ylabel("Accuracy (normalized sum of metrics)")
    ax.set_title(f"Accuracy vs Speed ({acc_title})")
    ax.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


# ----------------------------
# Orchestration
# ----------------------------


def build_dirs(base_out: Path) -> Dict[str, Path]:
    d = {
        "main_tables": base_out / "main" / "tables",
        "main_plots_lines": base_out / "main" / "plots" / "lines",
        "main_plots_aggregated": base_out / "main" / "plots" / "bar",
        "appendix_tables": base_out / "appendix" / "tables",
        "appendix_plots_lines": base_out / "appendix" / "plots" / "lines",
    }
    for p in d.values():
        _ensure_dir(p)
    return d


def run_pipeline(
    base_dir: Path,
    dir_prefix: str,
    out_dir: Path,
    lengthscales: Sequence[int],
    grid_sizes: Optional[Sequence[int]],
    plot_metrics: Sequence[str],
    main_table_metrics: Sequence[str],
    appendix_table_metrics: Sequence[str],
    exclude_models: Sequence[str],
    highlight_models: Optional[Sequence[str]],
    line_or_bar: str = "line",
):
    df = load_all_results(base_dir, dir_prefix, lengthscales)
    df = ensure_grid_sizes(df, grid_sizes)
    df = df[~df.model_name.isin(exclude_models)].reset_index(drop=True)
    df.num_chains[df.model_name == "ADVI"] = 4
    for col in df.columns:
        if "ess" in col.lower() or col.lower() == "infer_time":
            df[col] = df[col] / df["num_chains"]
    for col in df.columns:
        if "ess" in col.lower():
            df[col + " per second"] = df[col] / df["infer_time"]
    dirs = build_dirs(out_dir)

    # ----------------------------
    # Tables (main tables use main_table_metrics)
    # ----------------------------
    top2, top2_agg_ls, top2_agg_grid = make_best_model_table(
        df, main_table_metrics, topk=2
    )
    export_table(
        top2,
        dirs["main_tables"] / "top2_per_metric.csv",
        dirs["main_tables"] / "top2_per_metric.tex",
    )
    export_table(
        top2_agg_ls,
        dirs["main_tables"] / "top2_agg_ls_per_metric.csv",
        dirs["main_tables"] / "top2_agg_ls_per_metric.tex",
    )
    export_table(
        top2_agg_grid,
        dirs["main_tables"] / "top2_agg_grid_per_metric.csv",
        dirs["main_tables"] / "top2_agg_grid_per_metric.tex",
    )
    # ----------------------------
    # Aggregated tables (mean ± std with bolded best)
    # ----------------------------
    agg_grid = make_aggregated_model_table(
        df, main_table_metrics + ["train_time"], groupby="grid_size"
    )
    export_table(
        agg_grid,
        dirs["main_tables"] / "agg_grid_per_metric.csv",
        dirs["main_tables"] / "agg_grid_per_metric.tex",
    )

    # ----------------------------
    # Main plots (use plot_metrics)
    # ----------------------------
    style = PlotStyle(line_or_bar)
    plot_lines_or_bars_for_metrics(
        df, plot_metrics, dirs["main_plots_lines"], highlight_models, style
    )
    for s_metric in plot_metrics:
        plot_aggregated_bar(
            df, s_metric, dirs["main_plots_aggregated"] / f"{s_metric}.png"
        )
    plot_mean_infer_time(
        df, dirs["main_plots_lines"] / "mean_infer_time.png", highlight_models
    )
    plot_mean_infer_time(df, dirs["appendix_plots_lines"] / "mean_infer_time_all.png")

    # ----------------------------
    # Appendix tables (use appendix_table_metrics)
    # ----------------------------
    for ls in lengthscales:
        full_tab = per_lengthscale_full_table(df, ls, metrics=appendix_table_metrics)
        export_table(
            full_tab,
            dirs["appendix_tables"] / f"full_table_ls{ls}.csv",
            dirs["appendix_tables"] / f"full_table_ls{ls}.tex",
        )

    # ----------------------------
    # Appendix plots (reuse plot_metrics)
    # ----------------------------
    plot_lines_or_bars_for_metrics(
        df, plot_metrics, dirs["appendix_plots_lines"], None, style
    )
    plot_accuracy_vs_speed_quadrant(
        df,
        dirs["main_plots_lines"] / "quadrant_acc_vs_speed.png",
    )
    print(f"✅ Done. Outputs saved under: {out_dir.resolve()}")


# ----------------------------
# CLI
# ----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scalability analysis tables & plots.")
    parser.add_argument("--base_dir", type=Path, default=Path("results"))
    parser.add_argument("--dir_prefix", type=str, default="scalability_log_priors")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/scalability"))
    parser.add_argument("--lengthscales", type=int, nargs="+", default=[10, 30, 50])
    parser.add_argument(
        "--grid_sizes", type=int, nargs="*", default=[16**2, 24**2, 32**2, 48**2, 64**2]
    )
    parser.add_argument("--style", type=str, choices=["line", "bar"], default="line")
    parser.add_argument(
        "--highlight_models",
        type=str,
        nargs="*",
        default=[
            "Baseline_GP",
            "INLA",
            "DeepRV + gMLP adamw",
            "DeepRV + MLP",
            "Inducing Points Large",
        ],
    )
    parser.add_argument(
        "--exclude_models",
        type=str,
        nargs="*",
        default=[
            "DeepRV + gMLP",
            "Inducing Points",
            "DeepRV + gMLP kAttn",
            "INLA Dense",
        ],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # run_ablation()
    run_pipeline(
        args.base_dir,
        args.dir_prefix,
        args.out_dir,
        args.lengthscales,
        args.grid_sizes,
        DEFAULT_PLOT_METRICS,
        DEFAULT_MAIN_TABLE_METRICS,
        DEFAULT_APPENDIX_TABLE_METRICS,
        args.exclude_models,
        args.highlight_models,
        args.style,
    )


if __name__ == "__main__":
    main()
