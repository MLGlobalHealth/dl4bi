from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyDataverse.api import DataAccessApi

base_url = "https://dataverse.harvard.edu/"
dataset_id = "doi:10.7910/DVN/Z29FR0/FFDQI3"
cache_path = "./tmp/"


def coordinate_transform(
    long,
    lat,
):
    """
    Transform Lat-Long coordinates to local 2D coordinates for Kenya of order O(1).
    """
    # read lat-long
    s = gpd.points_from_xy(long, lat, crs="wgs84")
    # convert to 2d approximation for Kenya (in meters E, N)
    # this gives 6m accuracy according to https://epsg.io/21097
    s = s.to_crs("epsg:21097")

    # rescale to [0,1]
    l, u = -523492.03, 823852.53  # from https://epsg.io/21097
    s = np.stack([s.x, s.y], axis=-1)
    s = (s - l) / (u - l)
    # rescale to [-1, 1]
    s = s * 2 - 1

    return s


def prepare_data(force_redownload: bool = False):
    file: Path = Path(cache_path) / dataset_id.replace("/", "_")

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

    # only include point surveys by default
    df = df.query("AREATYPE=='Point'")

    # Kenya 2015
    df = df.query("COUNTRY=='Kenya' & YY==2015")
    print(f"Selected {len(df)} rows.")

    # s = coordinate_transform(df.Long, df.Lat)
    s = np.stack([df.Long, df.Lat], axis=-1)
    print(f"Transformed locations: shape {s.shape}, range [{s.min()}, {s.max()}]")

    # skip time for now
    # t = (df.YY * 12 + df.MM).to_numpy()
    # t -= t.min()
    # t = t[..., None]

    # s = np.hstack([s, t])

    N = df.Ex.to_numpy()
    Np = df.Pf.to_numpy()

    return s, Np, N
