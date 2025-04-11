from itertools import chain
from os import environ
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from shapely import MultiPolygon, Polygon

from benchmarks.disease_mapping.utils import cartesian_product

CACHE_DIR = Path(environ.get("CACHE_DIR", "tmp"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEG_TO_SEC = 3600

pd.options.mode.copy_on_write = True


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
    import rpy2.robjects as ro
    from rpy2.rinterface_lib.sexp import (
        NACharacterType,
        NAComplexType,
        NAIntegerType,
        NALogicalType,
        NARealType,
    )

    NA = (
        NACharacterType,
        NAComplexType,
        NAIntegerType,
        NALogicalType,
        NARealType,
    )

    df = gpd.GeoDataFrame()

    with ro.default_converter.context():
        convert = ro.conversion.get_conversion().rpy2py
        for c in rdf.colnames:
            match c:
                case "geometry":
                    geometry = rdf.rx2(c)
                    geometry = [multipolygon_r2py(x) for x in geometry]
                    df = df.set_geometry(geometry)
                case _:
                    series = rdf.rx2(c)
                    series = [
                        (pd.NA if isinstance(x, NA) else convert(x)) for x in series
                    ]
                    df[c] = series

    return df.convert_dtypes()


def get_shape(iso: str, region: str | None = None):
    iso = iso.upper()
    cache_path = CACHE_DIR / f"{iso}_shape.feather"

    if cache_path.exists():
        df = gpd.read_feather(cache_path)
    else:
        from rpy2.robjects.packages import importr

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
    sparse: bool = False,
) -> np.ndarray[np.float32] | tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
    """Get locations grid for a country.

    Args:
        iso: iso country code
        region: region name (optional)
        resolution: grid resolution in arc-seconds. 30 is ~1km at the equator. Defaults to 150.
        clip: whether to clip the grid to within the country borders. Defaults to True.
        sparse: whether to return two arrays `longs`, `lats`, defining the grid instead. Default to False. Incompatible with `clip`.
    Returns:
        N x 2 array of longitude, latitude pairs (in degrees).
    """
    # w, s, e, n

    assert (clip and sparse) is False, "clip and sparse are incompatible."
    iso = iso.upper()
    shape = get_shape(iso, region)

    bbox = np.array(shape.bounds)
    bbox *= DEG_TO_SEC
    w, s = bbox[:2] // res  # round down
    e, n = -((-bbox[2:]) // res)  # round up

    longs = np.arange(w, e) * res / DEG_TO_SEC
    lats = np.arange(s, n) * res / DEG_TO_SEC

    if sparse:
        return longs, lats

    points = cartesian_product(longs, lats)

    if clip:
        df = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*points.T, crs="EPSG:4326"))
        df = df.clip(shape)
        points = np.stack((df.geometry.x, df.geometry.y), axis=-1)

    return points


def get_population(iso: str, locations: np.array, res: int = 150):
    """
    Returns population in grid cells with centers given by locations (Long, Lat)
    and resolution res in arc-seconds.
    """
    m = res / 3600 / 2  # convert to degrees / 2
    file_path = CACHE_DIR / f"{iso}_population.tif"

    if not file_path.exists():
        # per-pixel population counts
        url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/2007/{iso.upper()}/{iso.lower()}_ppp_2007_1km_Aggregated_UNadj.tif"
        print(f"Downloading population data from {url}")
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download data from {url}")
        with open(file_path, "wb") as f:
            f.write(response.content)

    result = []
    with rasterio.open(file_path) as data:
        for x, y in locations:
            window = data.window(x - m, y - m, x + m, y + m)
            counts = data.read(window=window)
            # invalid values are set to < 0
            result.append(counts.sum(where=counts > 0))
    return np.array(result)


def get_survey_data(
    iso: str | None,
    region: str | None = None,
    query: str | None = None,
    res: int | None = 150,  # if not None round to grid of given res in seconds
):
    """
    Loads Pf survey data from malariaAtlas.

    Args:
        iso: iso country code
        region: region name (optional)
        query: query string to filter the data, available columns:
            'geometry', 'dhs_id', 'site_id', 'site_name', 'latitude', 'longitude',
            'rural_urban', 'country', 'country_id', 'continent_id', 'month_start',
            'year_start', 'month_end', 'year_end', 'lower_age', 'upper_age',
            'examined', 'positive', 'pr', 'species', 'method', 'rdt_type',
            'pcr_type', 'malaria_metrics_available', 'location_available',
            'permissions_info', 'citation1', 'citation2', 'citation3'
        res: if not None, round to grid of given resolution in seconds. Default 150.
    """
    cache_path: Path = CACHE_DIR / "Africa_prevalence.feather"

    if cache_path.exists():
        df = gpd.read_feather(cache_path)
    else:
        from rpy2.robjects.packages import importr

        ma = importr("malariaAtlas")
        # download the data
        rdf = ma.getPR(continent="Africa", species="Pf")
        # convert to pandas dataframe
        df = r_to_gpd(rdf)
        df = df.dropna(subset=["longitude", "latitude", "positive", "examined"])
        df = df.set_geometry(
            gpd.points_from_xy(df.longitude, df.latitude, crs="EPSG:4326")
        )
        df.to_feather(cache_path)

    if iso is not None:
        iso = iso.upper()
        df = df.query("country_id==@iso")

    if region is not None:
        shape = get_shape(iso, region)
        df = df.clip(shape)

    if query:
        df = df.query(query)

    print(df)
    print(f"Selected {len(df)} rows.")
    assert len(df) > 0, "Number of surveys selected must be >0."

    if res is not None:
        lat = round_to_grid(df.geometry.x, res)
        lon = round_to_grid(df.geometry.y, res)
        df = df.set_geometry(gpd.points_from_xy(lon, lat))
        # merge surveys from points close-by into one
        df = df.groupby(
            # since skipping time in s, ignore time for merging
            ["geometry"],
            as_index=False,
        ).sum()
        # this is now nonsense except for the positive and examined columns

    # skip time for now
    # t = (df.YY * 12 + df.MM).to_numpy()
    # t -= t.min()
    # t = t[..., None]

    s = np.array([[p.x, p.y] for p in df.geometry])

    n_pos = df.positive.to_numpy()
    n = df.examined.to_numpy()

    return {"s": s, "n_pos": n_pos, "n": n}


def get_survey_data2(
    iso: str | None,
    region: str | None = None,
    query: str | None = None,
    res: int | None = 150,  # if not None round to grid of given res in seconds
):
    dataset_id = "doi:10.7910/DVN/Z29FR0/FFDQI3"
    url = f"https://dataverse.harvard.edu/api/access/datafile/:persistentId/?persistentId={dataset_id}"

    file_path: Path = CACHE_DIR / (dataset_id.replace("/", "_") + ".csv")

    if file_path.exists():
        print("Reading survey data from cache.")
        pass
    else:
        print(f"Downloading survey data from {url}.")
        response = requests.get(url, headers={"User-Agent": "pydataverse"})
        if response.status_code != 200:
            raise ValueError(f"Failed to download data from {url}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(response.content)

    df = pd.read_csv(file_path, sep="\t")

    # only include point surveys
    df = df.query("AREATYPE=='Point'")

    if iso is not None:
        iso = iso.upper()
        df = df.query("AFRADMIN2Code.str.startswith(@iso)")

    # region
    if region is not None:
        df = df.query("AFRADMIN2Code==@region")

    if query:
        df = df.query(query)

    print(df)
    print(f"Selected {len(df)} rows.")
    assert len(df) > 0, "Number of surveys selected must be >0."

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

    return {"s": s, "n_pos": n_pos, "n": n}
