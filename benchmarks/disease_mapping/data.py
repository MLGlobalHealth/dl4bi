from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pyDataverse.api import DataAccessApi
from rpy2.robjects.packages import importr
from shapely import MultiPolygon, Polygon

base_url = "https://dataverse.harvard.edu/"
dataset_id = "doi:10.7910/DVN/Z29FR0/FFDQI3"
cache_path = "./tmp/"

ma = importr("malariaAtlas")


DEG_TO_SEC = 3600


def cartesian_product(*xs):
    n = len(xs)
    return np.stack(np.meshgrid(*xs), axis=-1).reshape(-1, n)


def round_to_grid(x, res):
    """Rounds to the nearest multiple of m."""
    assert res > 0
    k, rem = divmod(x, res)
    if rem * 2 < res:
        return k * res
    else:
        return (k + 1) * res


def get_grid(
    iso: str,
    res: int = 150,
    exclude_outer: bool = True,
):
    """Get locations grid for a country.

    Args:
        iso: country code
        resolution: grid resolution in arc-seconds. 30 is ~1km at the equator. Defaults to 150.
        exclude_outer: whether to exclude points outside the country borders. Defaults to True.
        sparse: whether to output
    Returns:
        N x 2 array of longitude, latitude pairs (in degrees).
    """
    # w, s, e, n
    shape = ma.getShp(ISO=iso)
    bbox = np.array(ma.getSpBbox(shape))

    bbox *= DEG_TO_SEC
    w, s = bbox[:, 0] // res  # round down
    e, n = -((-bbox[:, 1]) // res)  # round up

    longs = np.arange(w, e) * res / DEG_TO_SEC
    lats = np.arange(s, n) * res / DEG_TO_SEC

    points = cartesian_product(longs, lats)
    if not exclude_outer:
        return points

    polygons = [np.array(x).squeeze(0) for x in shape.rx2("geometry")[0]]
    polygons = [Polygon(x) for x in polygons]
    multipolygon = MultiPolygon(polygons)

    df = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*points.T))
    df = df.clip(multipolygon)
    return np.stack((df.geometry.x, df.geometry.y), axis=-1)


def get_survey_data(cfg: DictConfig):
    file: Path = Path(cache_path) / dataset_id.replace("/", "_")

    if file.exists() and cfg.force_redownload is False:
        pass
    else:
        data_api = DataAccessApi(base_url)
        response = data_api.get_datafile(dataset_id)
        assert response.is_success, (
            f"Download failed, got response {response.status_code} with content {response.content.decode()}"
        )
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(response.content)

    df = pd.read_csv(file, sep="\t")

    # only include point surveys by default
    df = df.query("AREATYPE=='Point'")
    df = df.query("COUNTRY==@cfg.country")

    month, year = cfg.get("month"), cfg.get("year")
    assert not (year is None and month is not None), (
        "If year is unspecified so must be month."
    )
    if year is not None:
        df = df.query("YY==@year")
    if month is not None:
        df = df.query("MM==@month")
    print(f"Selected {len(df)} rows.")
    assert len(df) > 0, "Number of observations must be >0."

    # s = coordinate_transform(df.Long, df.Lat)
    s = np.stack([df.Long, df.Lat], axis=-1)
    print(f"Locations: shape {s.shape}, range [{s.min()}, {s.max()}]")

    # skip time for now
    # t = (df.YY * 12 + df.MM).to_numpy()
    # t -= t.min()
    # t = t[..., None]

    # s = np.hstack([s, t])

    N = df.Ex.to_numpy()
    Np = df.Pf.to_numpy()

    return s, Np, N
