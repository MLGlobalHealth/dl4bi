import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely import Polygon, Point

from benchmarks.disease_mapping.data import round_to_grid


def point_to_square(
    point: tuple[float, float],  # (Long, Lat)
    side_length: float,  # side length in degrees
):
    x, y = point
    m = side_length / 2
    return Polygon(
        [
            (x - m, y - m),
            (x + m, y - m),
            (x + m, y + m),
            (x - m, y + m),
            (x - m, y - m),
        ]
    )


def infer_resolution(s):
    lon, lat = s.T
    lon = np.unique(lon)
    lat = np.unique(lat)
    lon_diff = np.abs(lon[1:] - lon[:-1])
    lat_diff = np.abs(lat[1:] - lat[:-1])
    return min(lon_diff.min(), lat_diff.min())


def plot(
    s: np.ndarray,  # [S, 2], assuming from a grid
    samples: np.ndarray,  # [N, S]
):
    N, S = samples.shape
    assert s.shape == (S, 2)

    res = infer_resolution(s)

    mean = samples.mean(axis=0)
    geometry = [point_to_square(x, res) for x in s]

    df = gpd.GeoDataFrame.from_dict(
        {"mean": mean, "geometry": geometry},
        geometry="geometry",
    )

    return df.plot(
        column="mean",
        cmap="viridis",
        legend=True,
        legend_kwds={"label": "Prevalence"},
    )
