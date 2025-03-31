from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyDataverse.api import DataAccessApi

base_url = "https://dataverse.harvard.edu/"
dataset_id = "doi:10.7910/DVN/Z29FR0/FFDQI3"
cache_path = "./tmp/"


def coordinate_transform(lat, long):
    """
    Transform Lat-Long coordinates to local 2D coordinates for Kenya of order O(1).
    """
    # read lat-long
    s = gpd.points_from_xy(lat, long, crs="EPSG:4326")
    # convert to 2d approximation for Kenya
    s = s.to_crs("EPSG:21097")

    s = np.stack([s.x, s.y], axis=-1)

    scaling = 1 / 5e6
    return s * scaling


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
        file.write_bytes(response.content)

    df = pd.read_csv(file, sep="\t")

    # only include point surveys by default
    df = df.query("AREATYPE=='Point'")

    # Kenya 2015
    df = df.query("COUNTRY=='Kenya' & YY==2015")
    print(f"Selected {len(df)} rows.")

    s = coordinate_transform(df.Lat, df.Long)
    print(s.shape, s.min(), s.max())

    # skip time for now
    # t = (df.YY * 12 + df.MM).to_numpy()
    # t -= t.min()
    # t = t[..., None]

    # s = np.hstack([s, t])

    N = df.Ex.to_numpy()
    Np = df.Pf.to_numpy()

    return s, Np, N


if __name__ == "__main__":
    prepare_data()
