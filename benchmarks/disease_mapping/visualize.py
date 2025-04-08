import matplotlib.pyplot as plt
import numpy as np
from pyparsing import line
from scipy.datasets import face
from shapely import MultiPolygon
from shapely.plotting import plot_polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable


def infer_resolution(s):
    lon, lat = s.T
    lon = np.sort(np.unique(lon))
    lat = np.sort(np.unique(lat))
    lon_diff = np.abs(lon[1:] - lon[:-1])
    lat_diff = np.abs(lat[1:] - lat[:-1])
    return min(lon_diff.min(), lat_diff.min())


def plot(
    s: np.ndarray,  # [S, 2], assuming from a grid
    samples: np.ndarray,  # [N, S]
    shape: MultiPolygon | None = None,
):
    N, S = samples.shape
    assert s.shape == (S, 2)
    lat, lon = s.T

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), layout="compressed")
    for ax in axes:
        if shape is not None:
            plot_polygon(
                shape,
                ax=ax,
                facecolor="none",
                edgecolor="black",
                add_points=False,
                linewidth=0.1,
            )

    axes[0].set_title("Mean")
    c = axes[0].scatter(lat, lon, c=samples.mean(0), cmap="viridis")
    plt.colorbar(c, ax=axes[0])

    axes[1].set_title("SD")
    c = axes[1].scatter(lat, lon, c=samples.std(0), cmap="viridis")
    plt.colorbar(c, ax=axes[1])

    for ax in axes:
        ax.set_aspect("equal")

    return fig
