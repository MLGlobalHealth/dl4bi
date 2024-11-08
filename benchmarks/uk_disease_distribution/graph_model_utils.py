#!/usr/bin/env python3

from typing import Union

import flax.linen as nn
import geopandas as gpd
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from omegaconf import DictConfig, OmegaConf
from prior_cvae import PriorCVAE
from scipy.spatial import distance_matrix
from sps.gp import GP
from sps.kernels import matern_3_2, periodic, rbf
from sps.priors import Prior

from dl4bi.core import *  # noqa: F403


def icar(batch_size, adj_mat, graph_model):
    """
    Generate data based on the Intrinsic Conditional Auto-Regressive (ICAR) model.

    Args:
        s (jax.Array): Locations or nodes for ICAR model.
        batch_size (int): Number of samples to generate.

    Returns:
        Tuple containing generated data for ICAR model.
    """
    num_nodes = adj_mat.shape[0]
    D = jnp.diag(adj_mat.sum(axis=1))  # Diagonal matrix of node degrees
    precision_matrix = D - adj_mat
    K = jnp.linalg.pinv(precision_matrix, hermitian=True)
    tau_sampler = instantiate(graph_model.tau)

    def icar_loader(rng):
        while True:
            rng, z_rng, tau_rng = random.split(rng, 3)
            tau = tau_sampler.sample(tau_rng)
            z = random.normal(z_rng, shape=(batch_size, num_nodes))
            f = cholesky(num_nodes, (1 / tau) * K, z)
            yield f, z, [tau]

    return icar_loader, ["tau"]


def car(batch_size, adj_mat, graph_model):
    """
    Generate data based on the Conditional Auto-Regressive (CAR) model.

    Args:
        rng (jax.Array): Random key for generating data.
        s (jax.Array): Locations or nodes for CAR model.
        batch_size (int): Number of samples to generate.

    Returns:
        Tuple containing generated data for CAR model.
    """
    num_nodes = adj_mat.shape[0]
    D = jnp.diag(adj_mat.sum(axis=1))  # Diagonal matrix of node degrees
    tau_sampler = instantiate(graph_model.tau)
    alpha_sampler = instantiate(graph_model.alpha)

    def car_loader(rng):
        while True:
            rng, z_rng, tau_rng, alpha_rng = random.split(rng, 4)
            tau = tau_sampler.sample(tau_rng)
            alpha = alpha_sampler.sample(alpha_rng)
            precision_matrix = D - (alpha * adj_mat)
            K = jnp.linalg.pinv(precision_matrix, hermitian=True)
            z = random.normal(z_rng, shape=(batch_size, num_nodes))
            f = cholesky(num_nodes, (1 / tau) * K, z)
            yield f, z, [tau, alpha]

    return car_loader, ["tau", "alpha"]


def bym(batch_size, adj_mat, graph_model):
    """
    Generate data based on the Besag-York-Mollié (BYM) model.

    Args:
        s (jax.Array): Locations or nodes for CAR model.
        batch_size (int): Number of samples to generate.

    Returns:
        Tuple containing generated data for CAR model.
    """
    num_nodes = adj_mat.shape[0]
    D = jnp.diag(adj_mat.sum(axis=1))  # Diagonal matrix of node degrees
    I_n = jnp.eye(N=num_nodes, dtype=adj_mat.dtype)
    tau_sampler = instantiate(graph_model.tau)
    alpha_sampler = instantiate(graph_model.alpha)
    v_sampler = instantiate(graph_model.v)

    def bym_loader(rng):
        while True:
            rng, z_rng, tau_rng, alpha_rng, v_rng = random.split(rng, 5)
            tau = tau_sampler.sample(tau_rng)
            alpha = alpha_sampler.sample(alpha_rng)
            v = v_sampler.sample(v_rng)
            R = jnp.linalg.pinv(D - (alpha * adj_mat), hermitian=True)
            z = random.normal(z_rng, shape=(batch_size, num_nodes))
            f = cholesky(num_nodes, (1 / v) * I_n + (1 / tau) * R, z)
            yield f, z, [tau, alpha, v]

    return bym_loader, ["tau", "alpha", "v"]


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


def cholesky(
    num_locations: int,
    K: jax.Array,
    z: jax.Array,  # [B, L]
    jitter: float = 1e-5,
):
    """Creates samples using Cholesky covariance factorization.

    Args:
        num_locations: Number of location to compose which determnines the
        shape of the covariance matrix
        K: Kernel (covariance) of the locations.
        z: A random vector used to generate samples.
        jitter: Noise added for numerical stability in Cholesky
            decomposition. Insufficiently large values will result
            in nan values.

    Returns:
        `Lz`: samples from the kernel combined with a random vector `z`.
    """
    L = jax.lax.linalg.cholesky(K + jitter * jnp.eye(num_locations))
    return jnp.einsum("ij,bj->bi", L, z)


def instantiate(d: Union[dict, DictConfig]):
    """Convenience function to instantiate an object from a config."""
    if isinstance(d, DictConfig):
        d = OmegaConf.to_container(d, resolve=True)
    if "cls" in d:
        cls, kwargs = d["cls"], d.get("kwargs", {})
        for k in kwargs:
            if k == "act_fn":
                kwargs[k] = getattr(nn, kwargs[k])
            elif isinstance(kwargs[k], dict):
                kwargs[k] = instantiate(kwargs[k])
        return globals()[cls](**kwargs)
    elif "func" in d:
        return eval(d["func"])
    return d
