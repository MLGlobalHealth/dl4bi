import shutil
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax.training import orbax_utils, train_state
from jax import jit, value_and_grad
from jax.scipy.stats import norm
from numpyro.distributions import *  # noqa: F403
from omegaconf import DictConfig, OmegaConf
from orbax.checkpoint import PyTreeCheckpointer

from inference_models import *  # noqa: F403
from models import *  # noqa: F403
from sps.kernels import matern_1_2, matern_3_2, matern_5_2, periodic, rbf


@flax.struct.dataclass
class TrainState(train_state.TrainState):
    # kwargs stores any extra information associated with training,
    # i.e. batch norm stats or fixed (random) projections
    kwargs: FrozenDict = FrozenDict({})


@dataclass
class Callback:
    fn: Callable
    interval: int  # apply every interval of train_num_steps


def generate_model_name(cfg: DictConfig):
    spatial_prior = cfg.inference_model.spatial_prior.func
    dec_name = cfg.model.kwargs.decoder.cls
    return cfg.get("name", f"{cfg.model.cls}_{dec_name}_{spatial_prior}")


def get_train_step(model_cfg: DictConfig, cond_names: list[str]):
    var_idx = None if "var" not in cond_names else cond_names.index("var")
    train_step = elbo_train_step
    if model_cfg.cls == "DeepRV":
        train_step = partial(deep_RV_train_step, var_idx=var_idx)
    elif model_cfg.cls == "PriorCVAE":
        train_step = prior_cvae_train_step
    return train_step


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


@partial(jax.jit, static_argnames=["var_idx"])
def deep_RV_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    var_idx: Optional[int] = None,
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

    def deep_RV_loss(params):
        f, z, conditionals = batch
        f_hat = state.apply_fn(
            {"params": params}, z, conditionals, **kwargs, rngs={"extra": rng}
        )
        mse_loss = optax.squared_error(f_hat.squeeze(), f.squeeze()).mean()
        if var_idx is not None:
            var = conditionals[var_idx].squeeze()
            mse_loss = (1 / var) * mse_loss
        return mse_loss

    loss, grads = value_and_grad(deep_RV_loss)(state.params)
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

    def prior_cvae_loss(params):
        f, _, conditionals = batch
        f_hat, z_mu, z_std = state.apply_fn(
            {"params": params}, f, conditionals, **kwargs, rngs={"extra": rng}
        )
        kl_div = -jnp.log(z_std) + (z_std**2 + z_mu**2 - 1) / 2
        mse_loss = optax.squared_error(f_hat.squeeze(), f.squeeze()).mean()
        return (1 / (2 * 0.9)) * mse_loss + kl_div.mean()

    loss, grads = value_and_grad(prior_cvae_loss)(state.params)
    return state.apply_gradients(grads=grads), loss


def get_valid_step(model_cfg: DictConfig, cond_names: list[str]):
    ls_idx = None if "ls" not in cond_names else cond_names.index("ls")
    var_idx = None if "var" not in cond_names else cond_names.index("var")
    model_name = model_cfg.cls
    decoder_only = model_name == "DeepRV"

    def valid_step(rng, state, batch, prefix: str = "", **kwargs):
        f, z, conditionals = batch
        var = 1 if var_idx is None else conditionals[var_idx].squeeze()
        ls = None if ls_idx is None else conditionals[ls_idx].squeeze()
        params = {"params": state.params, **state.kwargs}
        rngs = {"extra": rng}
        z_mu, z_std = None, None
        f_hat = jit(state.apply_fn)(
            params, z if decoder_only else f, conditionals, **kwargs, rngs=rngs
        )
        if not decoder_only:
            f_hat, z_mu, z_std = f_hat
        mse_score = optax.squared_error(f_hat.squeeze(), f.squeeze()).mean()
        # NOTE: Normalizing the mse score with variance, aN(0,K)~N(0, a^2K), and
        norm_mse_score = (1 / var) * mse_score
        loss = norm_mse_score
        if not decoder_only:
            kl_div = -jnp.log(z_std) + (z_std**2 + z_mu**2 - 1) / 2
            logp = (
                (1 / (2 * 0.9)) * mse_score
                if model_name == "PriorCVAE"
                else -norm.logpdf(f, f_hat, 1.0).mean()
            )
            loss = logp + kl_div.mean()

        return {
            f"{prefix} loss": loss,
            f"{prefix} norm MSE": norm_mse_score,
            "ls": ls if ls is not None else None,
        }

    return valid_step


def set_nested_defaults(cfg, keys, value):
    current = cfg
    for key in keys[:-1]:  # Traverse to the second last key
        if not OmegaConf.is_dict(current[key]):
            current[key] = {}
        current = current[key]
    if not OmegaConf.is_dict(current[keys[-1]]):
        current[keys[-1]] = value


def build_model(model_cfg: DictConfig, s: jax.Array):
    """Instantiates a model and sets the default dimesions for
    the deepRV model for easier usage."""
    if model_cfg["cls"] in ["DeepRV", "PriorCVAE"]:
        original_struct = OmegaConf.is_struct(model_cfg)
        # NOTE: Temporarily disable struct mode
        OmegaConf.set_struct(model_cfg, False)
        if model_cfg["cls"] == "PriorCVAE":
            if "z_dim" not in model_cfg["kwargs"]:
                model_cfg["kwargs"]["z_dim"] = s.shape[0]
        for module in ["decoder", "encoder"]:
            if model_cfg["cls"] == "DeepRV" and module == "encoder":
                continue
            if module not in model_cfg["kwargs"]:
                model_cfg["kwargs"][module] = {}
            if "kwargs" not in model_cfg["kwargs"][module]:
                model_cfg["kwargs"][module]["kwargs"] = {}
            if "dims" not in model_cfg["kwargs"][module]["kwargs"]:
                model_cfg["kwargs"][module]["kwargs"]["dims"] = [
                    s.shape[0],
                    s.shape[0],
                ]
        OmegaConf.set_struct(model_cfg, original_struct)
    return instantiate(model_cfg)


def instantiate(d: Union[dict, DictConfig]):
    """Convenience function to instantiate objects config."""
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
    if "numpyro_dist" in d:  # Case for NumPyro distributions
        dist_cls, kwargs = d["numpyro_dist"], d.get("kwargs", {})
        kwargs = {k: jnp.array(i) for k, i in kwargs.items()}
        return globals()[dist_cls](**kwargs)
    elif "func" in d:
        return eval(d["func"])
    return d


def cosine_annealing_lr(
    num_steps: int = 100000,
    peak_lr: float = 1e-3,
    pct_warmup: float = 0.0,
    num_cycles: int = 1,
):
    """Create an n-cycle cosine annealing schedule."""
    n = num_steps // num_cycles
    sched = optax.cosine_onecycle_schedule(
        n,
        peak_lr,
        pct_warmup,
        div_factor=10,
        final_div_factor=10,
    )
    boundaries = n * jnp.arange(1, num_cycles)
    return optax.join_schedules([sched] * num_cycles, boundaries)


def save_ckpt(state: TrainState, cfg: DictConfig, path: Path):
    "Save a checkpoint."
    shutil.rmtree(path, ignore_errors=True)
    ckptr = PyTreeCheckpointer()
    ckpt = {"state": state, "config": OmegaConf.to_container(cfg, resolve=True)}
    save_args = orbax_utils.save_args_from_target(ckpt)
    ckptr.save(path.absolute(), ckpt, save_args=save_args)


def generate_surrogate_decoder(state: TrainState, model: nn.Module):
    """Wraps a VAE model to issue decoder only calls for sampling

    Args:
        state (TrainState): surrogate model

    Returns: the decoding function
    """

    @jax.jit
    def deep_rv_decoder(z, conditionals, **kwargs):
        return state.apply_fn(
            {"params": state.params, **state.kwargs}, z, conditionals, **kwargs
        )

    @jax.jit
    def priorCVAE_decoder(z, conditionals, **kwargs):
        return model.apply(
            {"params": state.params, **state.kwargs},
            z,
            conditionals,
            **kwargs,
            method="decode",
        )

    if model.__class__.__name__ == "DeepRV":
        return deep_rv_decoder
    return priorCVAE_decoder


# NOTE: placeholder prior functions, to allow similar initialization across all spatial priors
def car():
    pass


def iid():
    pass
