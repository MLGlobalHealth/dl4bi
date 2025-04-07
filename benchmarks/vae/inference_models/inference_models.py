from functools import partial
from typing import Callable

import geopandas as gpd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.util import cholesky_of_inverse
from omegaconf import DictConfig
from utils.map_utils import generate_adjacency_matrix


def poisson(
    cfg: DictConfig,
    s: jax.Array,
    map_data: gpd.GeoDataFrame,
    spatial_prior: Callable,
    priors: dict,
    surrogate_kwargs: dict = {},
):
    """
    Builds an inference model for both GP baseline and surrogate inference.

    Args:
        cfg: run configuration
        s: Locations (n, dim_s).
        map_data: geo pandas map information
        spatial_prior: spatial prior - either GP kernel function or placeholder function
        priors: Dictionary of prior distributions.

    Returns:
        A NumPyro model function.
    """
    spatial_prior_model, cond_names = gen_saptial_prior(
        cfg, s, spatial_prior, priors, map_data, surrogate_kwargs
    )

    def inference_model(surrogate_decoder=None, obs_mask=True, y=None):
        mu, _, conditionals = spatial_prior_model(surrogate_decoder=surrogate_decoder)
        beta = numpyro.sample("beta", priors["beta"], sample_shape=())
        lambda_ = jnp.exp(beta + mu)
        with numpyro.handlers.mask(mask=obs_mask):
            f = numpyro.sample("obs", dist.Poisson(rate=lambda_), obs=y)
        return f, mu, jnp.concat([conditionals, jnp.array([beta])])

    return inference_model, cond_names + ["beta"]


def binomial(
    cfg: DictConfig,
    s: jax.Array,
    map_data: gpd.GeoDataFrame,
    spatial_prior: Callable,
    priors: dict,
    surrogate_kwargs: dict = {},
):
    """
    Builds a Binomial inference model for either actual spatial prior or a surrogate.
    NOTE: The Binomial model expects a 'population' column in the dataframe
        for it to be able to sample from the binomial distribution

    Args:
        cfg: run configuration
        s: Locations (n, dim_s).
        map_data: geo pandas map information
        spatial_prior: spatial prior - either GP kernel function or placeholder function
        priors: Dictionary of prior distributions.

    Returns:
        A NumPyro model function.
    """
    if "population" not in map_data.columns:
        raise ValueError(
            "Missing population column in dataframe."
            "The Binomial model expects a 'population' column in the dataframe"
            "for it to be able to sample."
        )
    population = jnp.array(map_data["population"].values, dtype=jnp.int32)
    if cfg.inference_model.get("population_scale", None) is not None:
        population = population // cfg.inference_model.population_scale
    spatial_prior_model, cond_names = gen_saptial_prior(
        cfg, s, spatial_prior, priors, map_data, surrogate_kwargs
    )

    def inference_model(surrogate_decoder=None, obs_mask=True, y=None):
        mu, _, conditionals = spatial_prior_model(surrogate_decoder=surrogate_decoder)
        beta = numpyro.sample("beta", priors["beta"], sample_shape=())
        eta = mu + beta
        with numpyro.handlers.mask(mask=obs_mask):
            f = numpyro.sample("obs", dist.Binomial(population, logits=eta), obs=y)
        return f, mu, jnp.concat([conditionals, jnp.array([beta])])

    return inference_model, cond_names + ["beta"]


def noisy_spatial_model(
    cfg: DictConfig,
    s: jax.Array,
    map_data: gpd.GeoDataFrame,
    spatial_prior: Callable,
    priors: dict,
    surrogate_kwargs: dict = {},
):
    """
    Spatial effects only model with noise sigma

    Args:
        cfg: run configuration
        s: Locations (n, dim_s).
        map_data: geo pandas map information
        spatial_prior: spatial prior - either GP kernel function or placeholder function
        priors: Dictionary of prior distributions.

    Returns:
        A NumPyro model function.
    """
    spatial_prior_model, cond_names = gen_saptial_prior(
        cfg, s, spatial_prior, priors, map_data
    )

    def inference_model(surrogate_decoder=None, obs_mask=True, y=None):
        mu, _, conditionals = spatial_prior_model(
            surrogate_decoder=surrogate_decoder, surrogate_kwargs=surrogate_kwargs
        )
        sigma = numpyro.sample("sigma", priors["sigma"], sample_shape=())
        with numpyro.handlers.mask(mask=obs_mask):
            f = numpyro.sample("obs", dist.Normal(loc=mu, scale=sigma), obs=y)
        return f, mu, jnp.concat([conditionals, jnp.array([sigma])])

    return inference_model, cond_names + ["sigma"]


def gen_saptial_prior(
    cfg: DictConfig,
    s: jax.Array,
    spatial_prior: Callable,
    priors: dict,
    map_data,
    surrogate_kwargs: dict = {},
):
    """Samples a distance based GP sample, used to reduce code duplication

    Args:
        s (jax.Array): locations
        kernel: kernel function
        priors (dict): priors of the inference model
        I_jitter (jax.Array): diagonal matrix of added jitter to give numerical
            stability to the sampling process

    Returns:
        The spatial prior sampler, and conditional hyperparameters names
    """
    n = s.shape[0]
    spatial_name = spatial_prior.__name__
    if spatial_name == "car":
        if map_data is None:
            raise ValueError("CAR model isn't supported for grid data")
        adj_mat = generate_adjacency_matrix(map_data, cfg.data.graph_construction)
        D = jnp.diag(adj_mat.sum(axis=1))
        model = partial(
            car_model,
            n=n,
            priors=priors,
            adj_mat=adj_mat,
            D=D,
            surrogate_kwargs=surrogate_kwargs,
        )
        cond_names = ["tau", "alpha"]
    elif spatial_name == "iid":
        cond_names = ["var"]
        model = partial(
            iid_locations, priors=priors, n=n, surrogate_kwargs=surrogate_kwargs
        )
    elif spatial_name in ["rbf", "periodic", "matern_3_2", "matern_5_2", "matern_1_2"]:
        cond_names = ["var", "ls"] + (["period"] if spatial_name == "periodic" else [])
        model = partial(
            distance_gp,
            s=s,
            kernel=spatial_prior,
            priors=priors,
            surrogate_kwargs=surrogate_kwargs,
        )
    else:
        raise ValueError(f"Chosen kernel {spatial_name} is not supported for inference")
    return model, cond_names


def distance_gp(
    s: jax.Array,
    kernel,
    priors: dict,
    jitter: float = 5e-4,
    surrogate_decoder=None,
    batch_size=1,
    surrogate_kwargs={},
):
    variance = numpyro.sample("var", priors["var"]).squeeze()
    lengthscale = numpyro.sample("ls", priors["ls"]).squeeze()
    conditionals = [jnp.array(variance), jnp.array(lengthscale)]
    if kernel.__name__ == "periodic":
        conditionals += [numpyro.sample("period", priors["period"])]
    conditionals = jnp.array(conditionals)
    z = numpyro.sample("z", dist.Normal(), sample_shape=(batch_size, s.shape[0]))
    if surrogate_decoder is not None:
        mu = numpyro.deterministic(
            "mu", surrogate_decoder(z, conditionals, **surrogate_kwargs).squeeze()
        )
    else:
        K = kernel(s, s, *conditionals) + jitter * jnp.eye(s.shape[0])
        mu = numpyro.deterministic("mu", cholesky(s.shape[0], K, z, jitter).squeeze())
    return mu, z, conditionals


def car_model(
    n: int,
    priors: dict,
    adj_mat: jax.Array,
    D: jax.Array,
    surrogate_decoder=None,
    batch_size=1,
    surrogate_kwargs={},
):
    tau = numpyro.sample("tau", priors["tau"]).squeeze()
    alpha = numpyro.sample("alpha", priors["alpha"]).squeeze()
    conditionals = jnp.array([jnp.array(tau), jnp.array(alpha)])
    z = numpyro.sample("z", dist.Normal(), sample_shape=(batch_size, n))
    if surrogate_decoder is not None:
        mu = numpyro.deterministic(
            "mu", surrogate_decoder(z, conditionals, **surrogate_kwargs).squeeze()
        )
    else:
        precision_mat = tau * (D - (alpha * adj_mat))
        L = cholesky_of_inverse(precision_mat)
        mu = numpyro.deterministic("mu", jnp.einsum("ij,bj->bi", L, z).squeeze())
    return mu, z, jnp.array(conditionals)


def iid_locations(priors, n, surrogate_decoder=None, surrogate_kwargs={}):
    var = numpyro.sample("var", priors["var"], sample_shape=()).squeeze()
    conditionals = jnp.array([var])
    z = numpyro.sample("z", dist.Normal(), sample_shape=(n,))
    if surrogate_decoder is not None:
        mu = numpyro.deterministic(
            "mu", surrogate_decoder(z, conditionals, **surrogate_kwargs).squeeze()
        )
    else:
        mu = numpyro.deterministic("mu", var * z)
    return mu, z, conditionals


def cholesky(num_locations: int, K: jax.Array, z: jax.Array, jitter: float = 1e-4):
    """Creates samples using Cholesky covariance factorization.

    Args:
        num_locations: Number of location
        K: Kernel (covariance) matrix.
        z: A random vector used to generate samples.
        jitter: Noise added for numerical stability in Cholesky
            decomposition. Insufficiently large values will result
            in nan values.

    Returns:
        `Lz`: samples from the kernel combined with a random vector `z`.
    """
    L = jnp.linalg.cholesky(K + jitter * jnp.eye(num_locations))
    return jnp.einsum("ij,bj->bi", L, z)
