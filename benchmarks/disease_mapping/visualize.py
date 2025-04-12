import matplotlib.pyplot as plt
import numpy as np
from shapely import MultiPolygon
from shapely.plotting import plot_polygon


def infer_resolution(s):
    lon, lat = s.T
    lon = np.sort(np.unique(lon))
    lat = np.sort(np.unique(lat))
    lon_diff = np.abs(lon[1:] - lon[:-1])
    lat_diff = np.abs(lat[1:] - lat[:-1])
    return min(lon_diff.min(), lat_diff.min())


def scatter_map(
    locations, values, ax: plt.Axes | None = None, cmap="viridis", vmin=None, vmax=None
):
    """
    Scatter plot with a colorbar.
    """
    if ax is None:
        ax = plt.axes()

    res = infer_resolution(locations)
    lon, lat = locations.T
    lon_range = np.arange(lon.min() - res / 2, lon.max() + res, res)
    lat_range = np.arange(lat.min() - res / 2, lat.max() + res, res)
    M, N = len(lon_range) - 1, len(lat_range) - 1
    X, Y = np.meshgrid(lon_range, lat_range)

    lon_binned = np.digitize(lon, lon_range) - 1
    lat_binned = np.digitize(lat, lat_range) - 1

    data = np.full((N, M), np.nan)
    for i, j, v in zip(lon_binned, lat_binned, values):
        data[j, i] = v

    ax.set_xlim(lon.min() - 0.5, lon.max() + 0.5)
    ax.set_ylim(lat.min() - 0.5, lat.max() + 0.5)
    sm = ax.pcolormesh(X, Y, data, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    plt.colorbar(sm, ax=ax)

    return ax


def plot_surveys_ax(data, ax=None):
    s, n_pos, n = data["s"], data["n_pos"], data["n"]

    if ax is None:
        ax = plt.Axes()

    scatter = ax.scatter(
        *s.T,
        c=n_pos / n,
        s=n,
        edgecolors="black",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax.legend(
        *scatter.legend_elements("sizes", num=[10, 20, 50, 100, 200]),
        title="Survey size",
    )
    ax.set_aspect("equal")
    ax.set_title("Surveys")
    plt.colorbar(scatter, ax=ax, label="Fraction of positive tests")

    return ax


def plot_surveys(
    data,
    shape: MultiPolygon | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 10), layout="compressed")
    ax = plot_surveys_ax(data, shape, ax)

    if shape is not None:
        plot_polygon(
            shape,
            ax=ax,
            facecolor="none",
            edgecolor="black",
            add_points=False,
            linewidth=0.5,
        )

    return fig


def plot_predictions(
    s: np.ndarray,  # [N, S, 2] or [S, 2], assuming from a grid
    theta: np.ndarray,  # [N, S]
    shape: MultiPolygon | None = None,
    data: np.ndarray | None = None,
):
    match s.shape:
        case (N, S, 2):
            s = s[0]  # assuming the locations are repeated
            assert theta.shape == (N, S)
        case (S, 2):
            N, _S = theta.shape
            assert _S == S
        case _:
            raise ValueError(f"Invalid shape {s.shape}. Expected (N, S, 2) or (S, 2).")

    fig, axes = plt.subplots(
        1,
        3 if data is None else 4,
        figsize=(30, 10),
        layout="compressed",
    )

    for ax in axes.flat:
        ax.set_aspect("equal")
        if shape is not None:
            plot_polygon(
                shape,
                ax=ax,
                facecolor="none",
                edgecolor="black",
                add_points=False,
                linewidth=0.5,
            )

    if data is not None:
        ax, *axes = axes
        plot_surveys_ax(data, ax=ax)
        ax.set_xlim(s[:, 0].min() - 0.5, s[:, 0].max() + 0.5)
        ax.set_ylim(s[:, 1].min() - 0.5, s[:, 1].max() + 0.5)

    scatter_map(s, theta.mean(0), ax=axes[0], vmin=0, vmax=1)
    scatter_map(s, theta.std(0), ax=axes[1], vmin=0, vmax=1)
    scatter_map(s, theta.std(0) / theta.mean(0), ax=axes[2], vmin=0, vmax=None)
    axes[0].set_title("Mean")
    axes[1].set_title("SD")
    axes[2].set_title("RSD")
    fig.suptitle("Prevalence predictions")

    return fig


def plot_distribution(samples):
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)

    fig, ax = plt.subplots(layout="compressed")
    ax.hist(samples, bins="auto")

    custom_preamble = {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
    plt.rcParams.update(custom_preamble)

    std = f"{std:.3f}"
    if mean < 0:
        std = r"\phantom{-}" + std
    mean = f"{mean:.3f}"

    text = rf"\mu &= {mean} \\ \sigma &= {std}"
    text = r"\begin{align*}" + text + r"\end{align*}"

    ax.text(
        0.95,
        0.95,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=14,
    )
    return fig
