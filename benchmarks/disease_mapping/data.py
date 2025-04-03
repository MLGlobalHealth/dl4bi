from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pyDataverse.api import DataAccessApi

base_url = "https://dataverse.harvard.edu/"
dataset_id = "doi:10.7910/DVN/Z29FR0/FFDQI3"
cache_path = "./tmp/"


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
