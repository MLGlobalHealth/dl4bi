import os
import zipfile
from urllib import request

import geopandas as gpd
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from omegaconf import DictConfig
from shapely import MultiPolygon, Point, unary_union
from shapely.affinity import scale, translate

MAPS_DATA_PATH = "benchmarks/uk_disease_distribution/maps/{map_name}/{data_type}"
MAP_REMOTE_PATH = {
    "England": "https://drive.google.com/uc?export=download&id=1qo5KyZV2v9gMl77ZB6waaCOjr-hBZkjQ&confirm=t"
}


def download_and_extract_map(map_name, raw_path):
    # NOTE if downloads fail due to file size\virus scan, then
    # Manually download from the link above
    if not os.path.exists(raw_path):
        if not os.path.exists(os.path.dirname(raw_path)):
            os.makedirs(os.path.dirname(raw_path))
        zip_url = MAP_REMOTE_PATH[map_name]
        zip_path = f"benchmarks/uk_disease_distribution/maps/{map_name}.zip"
        print(f"Downloading {map_name} from {zip_url}...")

        request.urlretrieve(zip_url, zip_path)

        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(f"benchmarks/uk_disease_distribution/maps/{map_name}")
        print("Unzipped to benchmarks/uk_disease_distribution/maps/")

        os.remove(zip_path)
        print(f"Deleted {zip_path}")


def get_raw_map_data(map_name):
    raw_path = MAPS_DATA_PATH.format(data_type="raw", map_name=map_name)
    download_and_extract_map(map_name, raw_path)
    return gpd.read_file(raw_path)


def normalize_geometry(gdf: gpd.GeoDataFrame):
    (x_trans, x_div), (y_trans, y_div) = get_norm_vars(gdf)

    def normalize_geometry(geom):
        centered_geom = translate(geom, xoff=-x_trans, yoff=-y_trans)
        normalized_geom = scale(
            centered_geom, xfact=1 / x_div, yfact=1 / y_div, origin=(0, 0)
        )
        return normalized_geom

    gdf["geometry"] = gdf.geometry.apply(normalize_geometry)
    return gdf


def prepare_coords_to_df(coords: ArrayLike, gdf: gpd.GeoDataFrame):
    df = pd.DataFrame({"coords": list(zip(coords[:, 0], coords[:, 1]))})
    df["coords"] = df["coords"].apply(Point)
    df = gpd.GeoDataFrame(df, geometry="coords", crs=gdf.crs)
    return df


def mask_out_of_poly(
    coords_gdf: gpd.GeoDataFrame, grid_points: ArrayLike, unified_geom: MultiPolygon
):
    # NOTE: First fast filter step with the convex hull
    mask = np.array(coords_gdf["coords"].within(unified_geom.convex_hull).values)
    grid_points = grid_points[mask]
    coords_gdf = coords_gdf[mask].reset_index(drop=True)
    mask = np.array(coords_gdf["coords"].within(unified_geom).values)
    grid_points = grid_points[mask]
    return grid_points


def create_uniform_grid(gdf: gpd.GeoDataFrame, data: DictConfig):
    gdf = normalize_geometry(gdf)
    # NOTE: Approximation of the actual map for faster contains checks
    unified_geom = gdf.geometry.convex_hull.union_all()

    minx, miny, maxx, maxy = unified_geom.bounds
    s_map = jnp.stack(
        [gdf.geometry.centroid.x.values, gdf.geometry.centroid.y.values], axis=-1
    )
    all_grid_points = []
    for density in data.sampling_grid.grid_densities:
        x_points = np.linspace(minx, maxx, density)
        y_points = np.linspace(miny, maxy, density)
        grid_x, grid_y = np.meshgrid(x_points, y_points)
        grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
        coords_gdf = prepare_coords_to_df(grid_points, gdf)
        grid_points = mask_out_of_poly(coords_gdf, grid_points, unified_geom)
        all_grid_points.extend(grid_points)
    all_grid_points = jnp.array(all_grid_points)
    processed_path = os.path.join(
        MAPS_DATA_PATH.format(data_type="processed", map_name=data.name), "data.npz"
    )
    jnp.savez(
        processed_path,
        all_grid_points=all_grid_points,
        s_map=s_map,
    )
    return all_grid_points, s_map


def process_map(data: DictConfig):
    map_name = data.name
    processed_path = os.path.join(
        MAPS_DATA_PATH.format(data_type="processed", map_name=map_name), "data.npz"
    )
    map_data = normalize_geometry(get_raw_map_data(map_name))
    bounds = jnp.array(
        [
            [map_data.bounds.minx.min(), map_data.bounds.maxx.max()],
            [map_data.bounds.miny.min(), map_data.bounds.maxy.max()],
        ]
    )
    all_map_data = dict(jnp.load(processed_path))
    sample_grid, s_map = all_map_data["all_grid_points"], all_map_data["s_map"]
    return s_map, sample_grid, bounds


def grid_valid_pct():
    num_samples = 1000
    for map_name in ["England"]:
        processed_path = os.path.join(
            MAPS_DATA_PATH.format(data_type="processed", map_name=map_name), "data.npz"
        )
        all_map_data = dict(jnp.load(processed_path))
        map_data = normalize_geometry(get_raw_map_data(map_name))
        sample_grid = all_map_data["all_grid_points"]
        sample_grid = sample_grid[
            np.random.choice(np.arange(len(sample_grid)), num_samples)
        ]
        unified_geom = unary_union(map_data.geometry)
        coords_gdf = prepare_coords_to_df(sample_grid, map_data)
        sample_grid = mask_out_of_poly(coords_gdf, sample_grid, unified_geom)
        print(
            f"{map_name} map approximation is {100 * len(sample_grid)/ num_samples:.2f}% valid"
        )


def get_norm_vars(gdf: gpd.GeoDataFrame):
    unified_geom = gdf.geometry.convex_hull.union_all()
    minx, miny, maxx, maxy = unified_geom.bounds
    return (minx, maxx), (miny, maxy)


if __name__ == "__main__":
    grid_valid_pct()
