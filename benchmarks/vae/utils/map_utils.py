import geopandas as gpd
import jax.numpy as jnp
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.spatial import distance_matrix
from shapely.affinity import scale, translate
from sps.utils import build_grid


def gen_locations(data: DictConfig):
    """Generates the map location information for training and inference.
    return map_data=None in case a grid is being used

    Args:
        data (DictConfig): data config

    Returns:
        map data, locations
    """
    map_data = (
        None if data.get("map_path", None) is None else gpd.read_file(data.map_path)
    )
    if map_data is None:
        s = build_grid(data.s).reshape(-1, len(data.s))  # flatten spatial dims
    else:
        s = process_map(map_data)
    return map_data, s


def normalize_geometry(gdf: gpd.GeoDataFrame):
    """Normalizes gdf geometry to 0-1 range, used to extract
    location centroids for learning

    Args:
        gdf (gpd.GeoDataFrame): geopandas data frame

    Returns:
        (gpd.GeoDataFrame): normalized geopandas data frame
    """
    (x_trans, x_div), (y_trans, y_div) = get_norm_vars(gdf)
    norm_gdf = gdf.copy()

    def norm_geom(geom):
        centered_geom = translate(geom, xoff=-x_trans, yoff=-y_trans)
        normalized_geom = scale(
            centered_geom, xfact=1 / x_div, yfact=1 / y_div, origin=(0, 0)
        )
        return normalized_geom

    norm_gdf["geometry"] = norm_gdf.geometry.apply(norm_geom)
    return norm_gdf


def process_map(gdf: gpd.GeoDataFrame):
    """Prepares map locations for the vae train or inference process

    Args:
        gdf (gpd.GeoDataFrame): geopandas data frame

    Returns:
        centroids locations of normalized map, the x,y bounds
    """
    map_data = normalize_geometry(gdf)
    centroids = map_data.geometry.centroid
    return jnp.stack([centroids.x.values, centroids.y.values], axis=-1)


def get_norm_vars(gdf: gpd.GeoDataFrame, s_max=100):
    """Returns the variables used to normalize the dataframe's geometries
    to a 0-1 range.

    Args:
        gdf (gpd.GeoDataFrame): geo dataframe

    Returns:
        x, y normalizing factors
    """
    centroids = gdf.geometry.centroid
    minx, maxx = centroids.x.min(), centroids.x.max()
    miny, maxy = centroids.y.min(), centroids.y.max()
    return (minx, (maxx - minx) / s_max), (miny, (maxy - miny) / s_max)


def generate_adjacency_matrix(gdf: gpd.GeoDataFrame, graph_construction: DictConfig):
    """
    Constructs an undirected adjacency matrix for a GeoDataFrame, where each (i, j) is 1
    if geometry i is adjacent to geometry j, and 0 otherwise. For isolated geoms
    the function connects the closest geom as a neighbor (by centroid distance).
    Parameters:
        gdf (gpd.GeoDataFrame): A GeoDataFrame with polygon geometries.

    Returns:
        jnp.array: A JAX array representing the adjacency matrix.
    """
    num_geoms = gdf.shape[0]
    adjacency_matrix = jnp.zeros((num_geoms, num_geoms), dtype=jnp.float32)

    for i, geom in enumerate(gdf.geometry):
        possible_neighbors = list(gdf.sindex.intersection(geom.bounds))

        for j in possible_neighbors:
            if i != j and geom.touches(gdf.geometry.iloc[j]):
                adjacency_matrix = adjacency_matrix.at[i, j].set(1.0)
                adjacency_matrix = adjacency_matrix.at[j, i].set(1.0)
    centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf.geometry])
    distances = distance_matrix(centroids, centroids)
    neighbor_sums = jnp.sum(adjacency_matrix, axis=1)
    isolated_indices = jnp.where(neighbor_sums == 0)[0]
    for i in isolated_indices:
        # NOTE: Exclude the diagonal to avoid self-distances
        distances[i, i] = np.inf
        closest_neighbor = jnp.argmin(distances[i])
        adjacency_matrix = adjacency_matrix.at[i, closest_neighbor].set(1.0)
        adjacency_matrix = adjacency_matrix.at[closest_neighbor, i].set(1.0)
    if graph_construction.self_loops:
        adjacency_matrix += jnp.eye(N=num_geoms, dtype=adjacency_matrix.dtype)
    return adjacency_matrix


def normalize_locations_names(s):
    """Normalize LTLA names for robust comparison."""
    if pd.isnull(s):
        return None
    return (
        str(s)
        .strip()  # Remove leading and trailing spaces
        .lower()  # Convert to lowercase
        .replace("-", " ")  # Replace dashes with spaces
        .replace("_", " ")  # Replace underscores with spaces
        .replace("  ", " ")  # Replace double spaces with single spaces
        .replace("city of london:westminster", "westminster")  # Specific edge cases
        .replace("cornwall:isles of scilly", "isles of scilly")
        .replace("rhondda cynon taff", "rhondda cynon taf")
    )


def area_data_to_geopandas(
    data_path: str,
    area_data_path: str,
    name_col: str,
    map_name: str,
    threshold: int = 100,
):
    from rapidfuzz import fuzz, process

    area_data = (
        pd.read_csv(data_path)
        if data_path.endswith("csv")
        else pd.read_excel(data_path)
    )
    all_area_geodata = gpd.read_file(area_data_path)
    all_area_geodata.rename(
        {c: "NAME" for c in all_area_geodata.columns if c.lower().endswith("nm")},
        axis=1,
        inplace=True,
    )
    all_area_geodata["normalized_name"] = all_area_geodata["NAME"].map(
        normalize_locations_names
    )
    area_data["normalized_name"] = area_data[name_col].map(normalize_locations_names)
    matches = area_data["normalized_name"].apply(
        lambda x: process.extractOne(
            x, all_area_geodata["normalized_name"], scorer=fuzz.ratio
        )
    )
    area_data["match_name"] = [match[0] if match else None for match in matches]
    area_data["match_score"] = [match[1] if match else None for match in matches]
    area_data["match_index"] = [match[2] if match else None for match in matches]
    manual_check_df = area_data[
        [name_col, "normalized_name", "match_name", "match_score"]
    ]
    unmatched = manual_check_df[manual_check_df["match_score"] < threshold]
    print(unmatched)
    area_data = area_data[area_data["match_score"] >= threshold].reset_index(drop=True)
    area_data = area_data.merge(
        all_area_geodata[["normalized_name", "geometry"]],
        left_on="match_index",
        right_index=True,
        how="left",
    )
    area_data.drop(
        columns=[
            "normalized_name_x",
            "normalized_name_y",
            "match_index",
            "match_name",
            "match_score",
        ],
        inplace=True,
    )
    area_geodata = gpd.GeoDataFrame(area_data, geometry="geometry")
    area_geodata.to_file(f"maps/{map_name}/raw/data.shp")
    # NOTE: to later perform manual checks and see that the matches fit
    manual_check_df.to_excel(f"maps/{map_name}/check.xlsx")


if __name__ == "__main__":
    name_c = "local authority: district / unitary (as of April 2021)"
    d_p = "maps/nomis_mortality/nomis_2025_01_06_091707.xlsx"
    a_d_p = "maps/Official_Divisions/Local_Authority_Districts_December_2021"
    area_data_to_geopandas(d_p, a_d_p, name_c, "nomis_mortality")
