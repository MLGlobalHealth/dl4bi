from typing import Optional, Union

import flax.linen as nn
import geopandas as gpd
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict
from flax.training import train_state
from jax import jit, random, value_and_grad
from jax.scipy.stats import norm
from models import *  # noqa: F403
from omegaconf import DictConfig, OmegaConf
from scipy.spatial import distance_matrix
from sps.gp import GP
from sps.kernels import l2_dist, matern_3_2, periodic, rbf
from sps.priors import Prior

from dl4bi.core import *  # noqa: F403


class TrainState(train_state.TrainState):
    kwargs: FrozenDict = FrozenDict({})


def generate_model_name(cfg: DictConfig, is_gp: bool, decoder_only: bool):
    return cfg.get(
        "name",
        f"VAE_{'GP' if is_gp else cfg.graph_model.name}_"
        f"{cfg.model.cls}_{cfg.model.kwargs.decoder.cls}_"
        f"{'dec' if decoder_only else ''}"
        f"{cfg.data.sampling_policy.replace('centroids', '')}",
    )


@jit
def elbo_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    **kwargs,
):
    """Standard VAE training step that uses an ELBO loss.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.

    Returns:
        `TrainState` with updated parameters.
    """

    def elbo_loss(params):
        f, _, conditionals = batch
        f_hat, z_mu, z_std = state.apply_fn(
            {"params": params}, f, conditionals, **kwargs, rngs={"extra": rng}
        )
        kl_div = -jnp.log(z_std) + (z_std**2 + z_mu**2 - 1) / 2
        logp = norm.logpdf(f, f_hat, 1.0).mean()
        return -logp + kl_div.mean()

    loss, grads = value_and_grad(elbo_loss)(state.params)
    return state.apply_gradients(grads=grads), loss


@jit
def mse_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    **kwargs,
):
    """A VAE decoder-only training step that uses an MSE loss.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.

    Returns:
        `TrainState` with updated parameters.
    """

    def mse_loss(params):
        f, z, conditionals = batch
        f_hat, _, _ = state.apply_fn(
            {"params": params}, z, conditionals, **kwargs, rngs={"extra": rng}
        )
        return optax.squared_error(f_hat.squeeze(), f.squeeze()).mean()

    loss, grads = value_and_grad(mse_loss)(state.params)
    return state.apply_gradients(grads=grads), loss


@jit
def prior_cvae_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    **kwargs,
):
    """The original PriorCVAE paper's train step.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.

    Returns:
        `TrainState` with updated parameters.
    """

    def elbo_loss(params):
        f, _, conditionals = batch
        f_hat, z_mu, z_std = state.apply_fn(
            {"params": params}, f, conditionals, **kwargs, rngs={"extra": rng}
        )
        kl_div = -jnp.log(z_std) + (z_std**2 + z_mu**2 - 1) / 2
        mse_loss = optax.squared_error(f_hat.squeeze(), f.squeeze()).mean()
        # TODO(Jhonathan): remove hard-coding
        # TODO change sigma -> check streching  on pre-violin plot
        return (1 / (2 * 0.9)) * mse_loss + kl_div.mean()

    loss, grads = value_and_grad(elbo_loss)(state.params)
    return state.apply_gradients(grads=grads), loss


def get_model_kwargs(
    s: jax.Array,
    preprocess: Optional[DictConfig],
    map_data: gpd.GeoDataFrame,
    graph_construction: DictConfig,
):
    model_kwargs, graph_data = {}, {}
    if preprocess is None:
        return model_kwargs
    bias, pos_enc = preprocess.get("bias", None), preprocess.get("pos_enc", None)
    if bias in ["effective_resistance"] or pos_enc is not None:
        graph_data["adjecency_matrix"] = generate_adjacency_matrix(
            map_data, graph_construction
        )
    if bias is not None:
        model_kwargs["bias"] = globals()[bias](s, s, graph_data)
    if pos_enc is not None:
        model_kwargs["pos_enc"] = globals()[pos_enc](s, graph_data)
    return model_kwargs


def dist_bias(s_ctx, s_test, graph_data):
    return l2_dist(s_ctx, s_test)


def effective_resistance(s_ctx, s_test, graph_data):
    A = graph_data["adjecency_matrix"]
    num_nodes = A.shape[0]
    L = jnp.fill_diagonal(jnp.zeros_like(A), jnp.sum(A, axis=1), inplace=False) - A
    ones = jnp.full((num_nodes, num_nodes), fill_value=1 / num_nodes)
    Gamma = jnp.linalg.pinv(L + ones, hermitian=True)
    diag_Gamma = jnp.diag(Gamma)
    return diag_Gamma[:, None] + diag_Gamma[None, :] - 2 * Gamma


def laplacian_pos_enc(s, graph_data, max_k=50):
    A = graph_data["adjecency_matrix"]
    num_nodes = A.shape[0]
    D = jnp.fill_diagonal(jnp.zeros_like(A), jnp.sum(A, axis=1), inplace=False)
    D_norm = jnp.pow(D, -0.5)
    L_norm = jnp.eye(num_nodes) - (D_norm @ A @ D_norm)
    evals, evects = jnp.linalg.eigh(L_norm)
    return jnp.concatenate([evals[..., None], evects], axis=-1)[..., :max_k]


def location_pos_enc(s, graph_data):
    return s.copy()


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

    def icar_loader(rng, bs=batch_size):
        while True:
            rng, z_rng, tau_rng = random.split(rng, 3)
            tau = tau_sampler.sample(tau_rng)
            z = random.normal(z_rng, shape=(bs, num_nodes))
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

    def car_loader(rng, bs=batch_size):
        while True:
            rng, z_rng, tau_rng, alpha_rng = random.split(rng, 4)
            tau = tau_sampler.sample(tau_rng)
            alpha = alpha_sampler.sample(alpha_rng)
            precision_matrix = D - (alpha * adj_mat)
            K = jnp.linalg.pinv(precision_matrix, hermitian=True)
            z = random.normal(z_rng, shape=(bs, num_nodes))
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

    def bym_loader(rng, bs=batch_size):
        while True:
            rng, z_rng, tau_rng, alpha_rng, v_rng = random.split(rng, 5)
            tau = tau_sampler.sample(tau_rng)
            alpha = alpha_sampler.sample(alpha_rng)
            v = v_sampler.sample(v_rng)
            R = jnp.linalg.pinv(D - (alpha * adj_mat), hermitian=True)
            z = random.normal(z_rng, shape=(bs, num_nodes))
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
