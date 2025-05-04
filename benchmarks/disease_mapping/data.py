from os import environ
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from shapely import MultiPolygon

from benchmarks.disease_mapping.utils import cartesian_product

pd.options.mode.copy_on_write = True

CACHE_DIR = Path(environ.get("CACHE_DIR", "tmp"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEG_TO_SEC = 3600

"""All countries where surveys were conducted."""
ALL_COUNTRIES = [
    'AGO', 'BDI', 'BEN', 'BFA', 'BWA', 'CAF', 'CIV', 'CMR', 'COD', 'COG', 'DJI', 
    'ERI', 'ETH', 'GAB', 'GHA', 'GIN', 'GMB', 'GNB', 'GNQ', 'KEN', 'LBR', 'MDG', 
    'MLI', 'MOZ', 'MRT', 'MWI', 'NAM', 'NER', 'NGA', 'RWA', 'SDN', 'SEN', 'SLE', 
    'SOM', 'SSD', 'SWZ', 'TCD', 'TGO', 'TZA', 'UGA', 'ZAF', 'ZMB', 'ZWE'
]  # fmt: skip


def round_to_multiple(x, m):
    """Rounds to the nearest multiple of `m`."""
    assert m > 0
    k, rem = divmod(x, m)

    floor = k * m
    return np.where(rem * 2 < m, floor, floor + m)


def round_to_grid(degrees, res):
    return round_to_multiple(degrees * DEG_TO_SEC, res) / DEG_TO_SEC


def get_shape(iso: str, region: str | None = None) -> MultiPolygon:
    iso = iso.upper()
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso}.gpkg"
    file_path = CACHE_DIR / f"{iso}.gpkg"

    if file_path.exists():
        print("Reading geodata from cache.")
    else:
        print(f"Downloading geodata from {url}.")
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download data from {url}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(response.content)

    if region is None:
        df = gpd.read_file(file_path, layer="ADM_ADM_0", engine="pyogrio")
    else:
        df = gpd.read_file(file_path, layer="ADM_ADM_1", engine="pyogrio")
        region_df = df.query("NAME_1==@region")

        if region_df.empty:
            available_regions = sorted(df["NAME_1"].dropna().unique())
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


def get_population(iso: str, locations: np.array, year=2007, res: int = 150):
    """
    Returns population in grid cells with centers given by locations (Long, Lat)
    and resolution res in arc-seconds.
    """
    year = int(year)
    iso = iso.upper()
    file_path = CACHE_DIR / f"{iso}_{year}_population.tif"
    assert 2000 <= year <= 2020

    if not file_path.exists():
        # per-pixel population counts
        url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/{year}/{iso}/{iso.lower()}_ppp_{year}_1km_Aggregated_UNadj.tif"
        print(f"Downloading population data from {url}")
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download data from {url}")
        with open(file_path, "wb") as f:
            f.write(response.content)

    result = []
    m = res / 3600 / 2  # offset from square's midpoint
    with rasterio.open(file_path) as data:
        for x, y in locations:
            window = data.window(x - m, y - m, x + m, y + m)
            counts = data.read(window=window)
            # invalid values are set to < 0
            result.append(counts.sum(where=counts > 0))
    return np.array(result).astype(np.float32)


def get_population_density(iso: str, locations: np.array, year=2007):
    """
    Returns population density at given points
    """
    year = int(year)
    iso = iso.upper()
    file_path = CACHE_DIR / f"{iso}_{year}_population_density.tif"
    # assert 2000 <= year <= 2020
    if year < 2000:
        year = 2000
        print(f"Year {year} is not available. Using 2000 instead.")
    if year > 2020:
        year = 2020
        print(f"Year {year} is not available. Using 2020 instead.")

    if not file_path.exists():
        url = f"https://data.worldpop.org/GIS/Population_Density/Global_2000_2020_1km_UNadj/{year}/{iso}/{iso.lower()}_pd_{year}_1km_UNadj.tif"
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download data from {url}")
        with open(file_path, "wb") as f:
            f.write(response.content)

    data: rasterio.DatasetReader
    with rasterio.open(file_path) as data:
        densities = list(data.sample(locations))
    return np.array(densities).squeeze(axis=-1).astype(np.float32)


def get_urban_rural(iso: str, locations: np.array, year=2007):
    densities = get_population_density(iso, locations, year)
    urban = densities >= 300
    rural = ~urban
    return np.stack([urban, rural], axis=-1).astype(np.float32)


def get_survey_data(
    iso: str | None,  # if None use all countries in Africa
    region: str | None = None,
    query: str | None = None,
    res: int | None = None,  # if not None round to grid of given res in seconds
    urban_rural: bool = False,
    time: bool = False,
):
    dataset_id = "doi:10.7910/DVN/Z29FR0/FFDQI3"
    url = f"https://dataverse.harvard.edu/api/access/datafile/:persistentId/?persistentId={dataset_id}"

    file_path: Path = CACHE_DIR / (dataset_id.replace("/", "_") + ".csv")

    if file_path.exists():
        print("Reading survey data from cache.")
        pass
    else:
        print(f"Downloading survey data from {url}.")
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download data from {url}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(response.content)

    df = pd.read_csv(file_path, sep="\t")

    # only include point surveys
    df = df.query("AREATYPE=='Point'")
    df["ISO"] = df["AFRADMIN2Code"].str[:3]

    if iso is not None:
        iso = iso.upper()
        df = df.query("ISO==@iso")

    # region
    if region is not None:
        df = df.query("AFRAdminname==@region")

    if query:
        df = df.query(query)

    print(df)
    print(f"Selected {len(df)} rows.")
    assert len(df) > 0, "Number of surveys selected must be >0."

    s = np.stack([df.Long, df.Lat], axis=-1)
    print(
        f"Locations: shape {s.shape}, bbox: ({s[:, 0].min()}, {s[:, 1].min()}), ({s[:, 0].max()}, {s[:, 1].max()})"
    )

    if urban_rural:
        x = np.zeros((s.shape[0], 2))
        for country in ALL_COUNTRIES:
            for year in df.YY.unique():
                mask = (df.ISO == country) & (df.YY == year)
                if mask.any():
                    print(f"Getting urban/rural for {country} in {year}")
                    urban_rural = get_urban_rural(country, s[mask], year)
                    x[mask] = urban_rural
    else:
        x = None

    if time:
        t = (df.YY + (df.MM - 1) / 12 - df.YY.min()).to_numpy()[..., None]
        print("Time extent:", t.min(), t.max())
        s = np.concat([s, t], axis=-1)

    n_pos = df.Pf.to_numpy()
    n = df.Ex.to_numpy()

    data = {"s": s, "n_pos": n_pos, "n": n}
    if x is not None:
        data["x"] = x
    return data
