import matplotlib.pyplot as plt
import numpy as np
from shapely import MultiPolygon
from shapely.plotting import plot_polygon

scatter_config = {
    # prevalence values are in [0,1]
    "vmin": 0,
    "vmax": 1,
    "cmap": "viridis",
}


def infer_resolution(s):
    lon, lat = s.T
    lon = np.sort(np.unique(lon))
    lat = np.sort(np.unique(lat))
    lon_diff = np.abs(lon[1:] - lon[:-1])
    lat_diff = np.abs(lat[1:] - lat[:-1])
    return min(lon_diff.min(), lat_diff.min())


def plot_surveys(
    s,  # [S, 2],
    n_pos,  # [S],
    n,  # [S],
    shape: MultiPolygon | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 10), layout="compressed")
    if shape is not None:
        plot_polygon(
            shape,
            ax=ax,
            facecolor="none",
            edgecolor="black",
            add_points=False,
            linewidth=0.5,
        )

    lat, lon = s.T
    scatter = ax.scatter(lat, lon, c=n_pos / n, s=n, **scatter_config)
    fig.colorbar(scatter, ax=ax, label="Fraction of positive tests")
    ax.legend(*scatter.legend_elements("sizes", alpha=0.6), title="Survey size")
    ax.set_aspect("equal")
    fig.suptitle("Surveys")
    return fig


def plot_predictions(
    s: np.ndarray,  # [N, S, 2] or [S, 2], assuming from a grid
    theta: np.ndarray,  # [N, S]
    shape: MultiPolygon | None = None,
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

    lat, lon = s.T

    fig, axes = plt.subplots(1, 3, figsize=(30, 10), layout="compressed")

    if shape is not None:
        for ax in axes:
            plot_polygon(
                shape,
                ax=ax,
                facecolor="none",
                edgecolor="black",
                add_points=False,
                linewidth=0.5,
            )

    axes[0].set_title("Mean")
    scatter = axes[0].scatter(lat, lon, c=theta.mean(0), **scatter_config)
    fig.colorbar(scatter, ax=axes[0])

    axes[1].set_title("SD")
    scatter = axes[1].scatter(lat, lon, c=theta.std(0), **scatter_config)
    fig.colorbar(scatter, ax=axes[1])

    axes[2].set_title("RSD")
    scatter = axes[2].scatter(
        lat,
        lon,
        c=theta.std(0) / theta.mean(0),
        **{
            **scatter_config,
            "vmax": None,  # RSD is unbounded
        },
    )
    fig.colorbar(scatter, ax=axes[2])

    fig.suptitle("Prevalence predictions")

    for ax in axes:
        ax.set_aspect("equal")

    return fig
