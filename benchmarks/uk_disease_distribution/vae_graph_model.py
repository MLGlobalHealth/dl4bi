#!/usr/bin/env python3
import pickle
from pathlib import Path
from typing import Optional

import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp

# import networkx as nx  # Import for graph-based data generation
import numpy as np
import optax
from jax import jit, random
from jax.scipy.stats import norm
from map_utils import get_raw_map_data, process_map
from omegaconf import DictConfig, OmegaConf
from plot_utils import log_vae_map_plots
from scipy.spatial import distance_matrix
from sps.priors import Prior
from tqdm import tqdm
from vae_gp import instantiate
from vae_train_utils import elbo_train_step, mse_train_step

import wandb
from dl4bi.core import *  # noqa: F403
from dl4bi.meta_regression.train_utils import (
    Callback,
    TrainState,
    cosine_annealing_lr,
    save_ckpt,
)
from dl4bi.vae import DeepChol, train_utils


@hydra.main("configs", config_name="default_vae_graph_model", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get(
        "name",
        f"VAE_{cfg.graph_model.name}_{cfg.model.cls}_{cfg.data.sampling_policy}",
    )
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))

    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    map_data = get_raw_map_data(cfg.data.name)
    model = instantiate(cfg.model)
    adj_mat = generate_adjacency_matrix(map_data)
    s, _, _ = process_map(cfg.data)
    dataloader, conditionals_names = {"car": car, "icar": icar, "bym": bym}[
        cfg.graph_model.name
    ](cfg.batch_size, adj_mat, cfg.graph_model)
    lr_schedule = cosine_annealing_lr(
        cfg.train_num_steps,
        cfg.lr_peak,
        cfg.lr_pct_warmup,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.clip_max_norm),
        optax.yogi(lr_schedule),
    )
    state = train(
        rng_train,
        dataloader,
        model,
        optimizer,
        isinstance(model, (DeepChol,)),
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[
            Callback(
                log_vae_map_plots(
                    map_data, s, conditionals_names, cfg.model.kwargs.z_dim
                ),
                cfg.plot_interval,
            )
        ],
    )
    results = validate(
        rng_test,
        dataloader,
        state,
        isinstance(model, (DeepChol,)),
        cfg.valid_num_steps,
        log_results=True,
    )
    path = Path(
        f"results/uk_disease_dist/{cfg.data.name}/{cfg.graph_model.name}/{cfg.seed}/{run_name}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".pkl"), "wb") as save_file:
        pickle.dump(results, save_file)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def train(
    rng: jax.Array,
    loader,
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    is_decoder_only: bool,
    train_num_steps: int = 100000,
    valid_num_steps: int = 25000,
    valid_interval: int = 25000,
    log_every_n: int = 100,
    callbacks: list[Callback] = [],
    state: Optional[TrainState] = None,
):
    rng_data, rng_params, rng_extra, rng_train = random.split(rng, 4)
    batches = loader(rng_data)
    f, z, conditionals = next(batches)
    rngs = {"params": rng_params, "extra": rng_extra}
    x = z if is_decoder_only else f
    train_step = mse_train_step if is_decoder_only else elbo_train_step
    kwargs = model.init(rngs, x, conditionals)
    params = kwargs.pop("params")
    param_count = nn.tabulate(model, rngs)(x, conditionals)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params if state is None else state.params,
        kwargs=kwargs if state is None else state.kwargs,
        tx=optimizer,
    )
    print(param_count)

    losses = np.zeros((train_num_steps,))
    for i in (pbar := tqdm(range(train_num_steps), unit="batch", dynamic_ncols=True)):
        rng_step, rng_train = random.split(rng_train)
        batch = next(batches)
        state, losses[i] = train_step(rng_step, state, batch)
        if (i + 1) % log_every_n == 0:
            avg = jnp.mean(losses[i - log_every_n : i])
            pbar.set_postfix(loss=f"{avg:.3f}")
            wandb.log({"loss": avg})
        if (i + 1) % valid_interval == 0:
            rng_valid, rng_train = random.split(rng_train)
            validate(
                rng_valid,
                loader,
                state,
                is_decoder_only,
                valid_num_steps,
            )
        for cbk in callbacks:
            if (i + 1) % cbk.interval == 0:
                cbk.fn(i, rng_step, state, batches, model)
    return state


def validate(
    rng: jax.Array,
    loader,
    state: train_utils.TrainState,
    is_decoder_only: bool,
    valid_num_steps: int = 5000,
    log_results: bool = False,
):
    rng_data, rng_extra = random.split(rng)
    losses = np.zeros((valid_num_steps,))
    batches = loader(rng_data)
    results = []
    for i in (_ := tqdm(range(valid_num_steps), unit="batch", dynamic_ncols=True)):
        batch = next(batches)
        f, z, conditionals = batch
        params = {"params": state.params, **state.kwargs}
        rngs = {"extra": rng_extra}
        if is_decoder_only:
            f_hat = jit(state.apply_fn)(params, z, conditionals)
            losses[i] = optax.squared_error(f_hat, f.squeeze()).mean()
        else:
            # NOTE: ELBO, assumes normal gaussian
            f_hat, z_mu, z_std = jit(state.apply_fn)(params, f, conditionals, rngs=rngs)
            kl_div = -jnp.log(z_std) + (z_std**2 + z_mu**2 - 1) / 2
            logp = norm.logpdf(f, f_hat, 1.0).mean()
            losses[i] = -logp + kl_div.mean()
        if log_results:
            b = [np.array(v) for v in batch]
            p = np.array(f_hat)
            results += [(b, p)]
    loss = losses.mean()
    print(f"validation loss: {loss:.3f}")
    wandb.log({"validation_loss": loss})
    return results


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
    K = jnp.linalg.pinv(precision_matrix)
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
            K = jnp.linalg.inv(precision_matrix)
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
            R = jnp.linalg.pinv(D - (alpha * adj_mat))
            z = random.normal(z_rng, shape=(batch_size, num_nodes))
            f = cholesky(num_nodes, (1 / v) * I_n + (1 / tau) * R, z)
            yield f, z, [tau, alpha, v]

    return bym_loader, ["tau", "alpha", "v"]


def generate_adjacency_matrix(gdf):
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


if __name__ == "__main__":
    main()
