from itertools import chain
from os import environ
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from pyDataverse.api import DataAccessApi
from rpy2.rinterface_lib.sexp import (
    NACharacterType,
    NAComplexType,
    NAIntegerType,
    NALogicalType,
    NARealType,
)
from rpy2.robjects.packages import importr
from shapely import MultiPolygon, Polygon, from_geojson, to_geojson

R_NA_VALUES = (NACharacterType, NAComplexType, NAIntegerType, NALogicalType, NARealType)


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


def multipolygon_r2py(geometry):
    geometry = chain.from_iterable(geometry)  # flatten
    geometry = map(np.array, geometry)
    geometry = map(Polygon, geometry)

    shape = MultiPolygon(geometry)
    return shape


def r_to_gpd(rdf):
    df = gpd.GeoDataFrame()

    with ro.default_converter.context():
        convert = ro.conversion.get_conversion().rpy2py
        for c in rdf.colnames:
            match c:
                case "geometry":
                    geometry = rdf.rx2(c)
                    geometry = [multipolygon_r2py(x) for x in geometry]
                    df[c] = geometry
                    df.set_geometry(c)
                case _:
                    series = rdf.rx2(c)
                    series = [
                        (pd.NA if isinstance(x, R_NA_VALUES) else convert(x))
                        for x in series
                    ]
                    df[c] = series
    return df.convert_dtypes()


def get_shape(iso: str, region: str | None = None):
    cache_path = CACHE_DIR / f"{iso}.feather"

    if cache_path.exists():
        df = gpd.read_feather(cache_path)
    else:
        df = r_to_gpd(
            importr("malariaAtlas").getShp(ISO=iso, admin_level=["admin0", "admin1"])
        )
        df.to_feather(cache_path)

    if region is not None:
        region_df = df.query("name_1==@region")
        if region_df.empty:
            available_regions = sorted(df["name_1"].dropna().unique())
            raise ValueError(
                f"Region not found. \nAvailiable regions: {available_regions}."
            )
        else:
            df = region_df

    return df["geometry"].iloc[0]


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
