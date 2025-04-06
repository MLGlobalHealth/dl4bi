from itertools import chain
from os import environ
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyDataverse.api import DataAccessApi
from shapely import MultiPolygon, Polygon, from_geojson, to_geojson

base_url = "https://dataverse.harvard.edu/"
dataset_id = "doi:10.7910/DVN/Z29FR0/FFDQI3"
CACHE_DIR = Path(environ.get("CACHE_DIR", "tmp"))


DEG_TO_SEC = 3600

pd.options.mode.copy_on_write = True


def cartesian_product(*xs):
    n = len(xs)
    return np.stack(np.meshgrid(*xs), axis=-1).reshape(-1, n)


def round_to_multiple(x, m):
    """Rounds to the nearest multiple of `m`."""
    assert m > 0
    k, rem = divmod(x, m)

    floor = k * m
    return np.where(rem * 2 < m, floor, floor + m)


def round_to_grid(degrees, res):
    return round_to_multiple(degrees * DEG_TO_SEC, res) / DEG_TO_SEC


def get_shape(iso: str, region: str | None = None) -> MultiPolygon:
    cache_path = iso + ("_" + region if region else "") + ".json"
    cache_path = CACHE_DIR / cache_path

    if cache_path.exists():
        print(f"Reading shape from cache: {cache_path}.")
        return from_geojson(cache_path.read_text())

    else:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr

        ma = importr("malariaAtlas")
        if region is None:
            shapes = ma.getShp(ISO=iso)
        else:
            shapes = ma.getShp(ISO=iso, admin_level="admin1")
            ro.r.assign("df", shapes)
            shapes = ro.r(f"subset(df, name_1=='{region}')")

        polygons = shapes.rx2("geometry")[0]
        polygons = chain.from_iterable(polygons)  # flatten
        polygons = map(np.array, polygons)
        polygons = map(Polygon, polygons)

        shape = MultiPolygon(polygons)
        cache_path.write_text(to_geojson(shape))

        return shape


def get_grid(
    iso: str,
    region: str | None = None,
    res: int = 150,
    clip: bool = True,
):
    """Get locations grid for a country.

    Args:
        iso: iso country code
        region: region name (optional)
        resolution: grid resolution in arc-seconds. 30 is ~1km at the equator. Defaults to 150.
        clip: whether to clip the grid to within the country borders. Defaults to True.
    Returns:
        N x 2 array of longitude, latitude pairs (in degrees).
    """
    # w, s, e, n

    shape = get_shape(iso, region)

    bbox = np.array(shape.bounds)
    bbox *= DEG_TO_SEC
    w, s = bbox[:2] // res  # round down
    e, n = -((-bbox[2:]) // res)  # round up

    longs = np.arange(w, e) * res / DEG_TO_SEC
    lats = np.arange(s, n) * res / DEG_TO_SEC

    points = cartesian_product(longs, lats)

    if clip:
        df = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*points.T))
        df = df.clip(shape)
        points = np.stack((df.geometry.x, df.geometry.y), axis=-1)

    return points


def get_survey_data(
    iso: str,
    year: int | None = None,
    month: int | None = None,
    res: int | None = 150,  # if not None round to grid of given res in seconds
    force_redownload=False,
):
    file: Path = CACHE_DIR / (dataset_id.replace("/", "_") + ".csv")

    if file.exists() and force_redownload is False:
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

    # only include point surveys
    df = df.query("AREATYPE=='Point'")
    df = df.query("AFRADMIN2Code.str.startswith(@iso)")

    assert not (year is None and month is not None), (
        "If year is unspecified so must be month."
    )
    if year is not None:
        df = df.query("YY==@year")
    if month is not None:
        df = df.query("MM==@month")
    print(f"Selected {len(df)} rows.")
    assert len(df) > 0, "Number of observations must be >0."

    if res is not None:
        df.Long = round_to_grid(df.Long, res)
        df.Lat = round_to_grid(df.Lat, res)
        # merge surveys from points close-by into one
        df = df.groupby(
            # since skipping time in s, ignore time for merging
            ["Lat", "Long"],
            # ["Lat", "Long", "YY", "MM"],
            as_index=False,
        ).sum()

    s = np.stack([df.Long, df.Lat], axis=-1)
    print(
        f"Locations: shape {s.shape}, bbox: ({s[:, 0].min()}, {s[:, 1].min()}), ({s[:, 0].max()}, {s[:, 1].max()})"
    )

    # skip time for now
    # t = (df.YY * 12 + df.MM).to_numpy()
    # t -= t.min()
    # t = t[..., None]

    # s = np.hstack([s, t])

    n_pos = df.Pf.to_numpy()
    n = df.Ex.to_numpy()

    return s, n_pos, n
