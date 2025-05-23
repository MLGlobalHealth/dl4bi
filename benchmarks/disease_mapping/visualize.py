import matplotlib.pyplot as plt
import numpy as np
from cartopy.crs import PlateCarree
from shapely import MultiPolygon

cmap = plt.get_cmap("viridis")
projection = PlateCarree()


def infer_resolution(s):
    lon, lat = s.T
    lon = np.sort(np.unique(lon))
    lat = np.sort(np.unique(lat))
    lon_diff = np.abs(lon[1:] - lon[:-1])
    lat_diff = np.abs(lat[1:] - lat[:-1])
    return min(lon_diff.min(), lat_diff.min())


def scatter_map(locations, values, ax: plt.Axes, cmap="viridis", vmin=None, vmax=None):
    """
    Scatter plot with a colorbar.
    """
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
    sm = ax.pcolormesh(X, Y, data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    plt.colorbar(sm, ax=ax)

    return ax


def plot_surveys_ax(data, ax: plt.Axes):
    s, n_pos, n = data["s"], data["n_pos"], data["n"]
    if s.shape[-1] == 3:
        # TODO draw time somehow?
        s = s[..., :2]

    scatter = ax.scatter(
        *s.T,
        c=n_pos / n,
        s=n,
        edgecolors="black",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )
    ax.legend(
        *scatter.legend_elements("sizes", num=[10, 20, 50, 100, 200]),
        title="Survey size",
    )
    ax.set_aspect("equal")
    ax.set_title("Surveys")
    ax.coastlines()
    ax.gridlines(draw_labels=True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Fraction of positive tests")


def plot_surveys(
    data,
):
    s = data["s"]
    if s.shape[-1] == 3:
        # TODO draw time somehow?
        s = s[..., :2]  # drop time
    fig, ax = map_grid(s, 1, 1)
    plot_surveys_ax(data, ax[0, 0])

    return fig


def map_grid(s, nrows, ncols, **kwargs):
    default_kwargs = dict(
        subplot_kw=dict(projection=projection),
        squeeze=False,
        layout="compressed",
        figsize=(ncols * 8, nrows * 8),
        sharex=True,
        sharey=True,
    )
    fig, axes = plt.subplots(nrows, ncols, **(default_kwargs | kwargs))
    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.coastlines()
        ax.gridlines(draw_labels=True, alpha=0.3)
        ax.set_xlim(s[:, 0].min() - 0.5, s[:, 0].max() + 0.5)
        ax.set_ylim(s[:, 1].min() - 0.5, s[:, 1].max() + 0.5)
    return fig, axes


def plot_predictions(
    s_t: np.ndarray,  # [N, S, 2] or [S, 2], assuming from a grid
    y_t: np.ndarray,  # [N, S],
    theta_t: np.ndarray,  # [N, S],
    data: np.ndarray | None = None,
    y_c: np.ndarray | None = None,
    shape: MultiPolygon | None = None,
):
    match s_t.shape:
        case (N, S, 2):
            s_t = s_t[0]  # assuming the locations are repeated
            assert theta_t.shape == (N, S)
        case (S, 2):
            N, _S = theta_t.shape
            assert _S == S
        case _:
            raise ValueError(
                f"Invalid shape {s_t.shape}. Expected (N, S, 2) or (S, 2)."
            )

    nrows = 2 if y_c is not None else 1
    ncols = 3
    fig, axes = map_grid(s_t, nrows, ncols)

    if data is not None:
        plot_surveys_ax(data, ax=axes[0, 0])

    scatter_map(s_t, theta_t.mean(0), ax=axes[0, 1], vmin=0, vmax=1)
    axes[0, 1].set_title("Predicted mean")
    scatter_map(s_t, theta_t.std(0), ax=axes[0, 2])
    axes[0, 2].set_title("Predicted SD")

    if y_c is not None:
        sm = axes[1, 0].scatter(
            *data["s"].T,
            c=y_c.mean(0),
            s=10 / y_c.std(0),  # size ~ 1 / std
        )
        plt.colorbar(sm, ax=axes[1, 0])
        axes[1, 0].set_title("Inferred spatial effect")
        scatter_map(s_t, y_t.mean(0), ax=axes[1, 1])
        axes[1, 1].set_title("Predicted spatial effect mean")
        scatter_map(s_t, y_t.std(0), ax=axes[1, 2])
        axes[1, 2].set_title("Predicted spatial effect SD")

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
