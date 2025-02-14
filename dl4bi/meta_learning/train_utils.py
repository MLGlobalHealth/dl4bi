import re
import shutil
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from flax.core import FrozenDict
from flax.training import orbax_utils, train_state
from jax import jit, random, vmap
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
from omegaconf import DictConfig, OmegaConf
from optax.losses import safe_softmax_cross_entropy
from orbax.checkpoint import PyTreeCheckpointer

# NOTE: needed by `instantiate` function
from sps.gp import GP
from sps.kernels import matern_1_2, matern_3_2, matern_5_2, periodic, rbf
from sps.priors import Prior
from sps.sir import LatticeSIR
from sps.utils import build_grid
from tqdm import tqdm

# NOTE: needed by `instantiate` function
from ..core.attention import *
from ..core.bias import *
from ..core.conv import *
from ..core.dist import *
from ..core.embed import *
from ..core.gnn import *
from ..core.knn import *
from ..core.metrics import *
from ..core.mlp import *
from ..core.preprocess import *
from ..core.transformer import *
from .anp import ANP
from .banp import BANP
from .bnp import BNP
from .canp import CANP
from .cnp import CNP
from .convcnp import ConvCNP
from .np import NP
from .sgnp import SGNP
from .tnp_d import TNPD
from .tnp_kr import TNPKR, ScanTNPKR
from .tnp_nd import TNPND
from .transform import *


@flax.struct.dataclass
class TrainState(train_state.TrainState):
    # kwargs stores any extra information associated with training,
    # i.e. batch norm stats or fixed (random) projections
    kwargs: FrozenDict = FrozenDict({})


@dataclass
class Callback:
    fn: Callable  # (step, rng_step, state, batch) -> None
    interval: int  # apply every interval of train_num_steps


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
        return globals().get(cls, getattr(nn, cls, None))(**kwargs)
    elif "func" in d:
        return eval(d["func"])
    return d


def train(
    rng: jax.Array,
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    train_step: Callable,
    valid_step: Callable,
    train_dataloader: Callable,
    valid_dataloader: Callable,
    train_num_steps: int = 100000,
    valid_num_steps: Optional[int] = None,
    valid_interval: int = 25000,
    log_loss_interval: int = 100,
    callbacks: list[Callback] = [],
    monitor_metric: str = "NLL",  # validation metric to monitor
    early_stop_patience: Optional[int] = None,
    state: Optional[TrainState] = None,
):
    rng_data, rng_params, rng_extra, rng_train = random.split(rng, 4)
    batches = train_dataloader(rng_data)
    s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, *_ = next(batches)
    rngs = {"params": rng_params, "extra": rng_extra}
    kwargs = model.init(rngs, s_ctx, f_ctx, s_test, valid_lens_ctx, valid_lens_test)
    params = kwargs.pop("params")
    param_count = nn.tabulate(model, rngs)(
        s_ctx,
        f_ctx,
        s_test,
        valid_lens_ctx,
        valid_lens_test,
    )
    print(param_count)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params if state is None else state.params,
        kwargs=kwargs if state is None else state.kwargs,
        tx=optimizer,
    )
    losses = []
    patience = 0
    best_state = state
    early_stop_patience = early_stop_patience or train_num_steps
    train_nll, valid_nll, best_metric = float("inf"), float("inf"), float("inf")
    pbar = tqdm(range(1, train_num_steps + 1), unit=" batches", dynamic_ncols=True)
    for i in pbar:
        batch = next(batches)
        rng_train_step, rng_train = random.split(rng_train)
        state, loss = train_step(rng_train_step, state, batch)
        losses += [loss]
        if i % log_loss_interval == 0:
            train_nll = np.mean(losses)
            losses = []
            wandb.log({"Train NLL": train_nll})
        if i % valid_interval == 0:
            rng_valid, rng_train = random.split(rng_train)
            metrics = evaluate(
                rng_valid,
                state,
                valid_step,
                valid_dataloader,
                valid_num_steps,
            )
            valid_nll = metrics["NLL"]
            metric = metrics[monitor_metric]
            wandb.log({f"Valid {m}": v for m, v in metrics.items()})
            patience += 1
            if metric < best_metric:
                patience = 0
                best_metric = metric
                best_state = state
            if patience >= early_stop_patience:
                return best_state
        for cbk in callbacks:
            if i % cbk.interval == 0:
                cbk.fn(i, rng_train_step, state, batch)
        pbar.set_postfix(
            {"Train NLL": f"{train_nll:.3f}", "Valid NLL": f"{valid_nll:.3f}"}
        )
    return best_state


def evaluate(
    rng: jax.Array,
    state: TrainState,
    valid_step: Callable,
    dataloader: Callable,
    num_steps: Optional[int],
):
    rng_data, rng = random.split(rng)
    num_steps = num_steps or float("inf")
    pbar = tqdm(
        dataloader(rng_data),
        total=num_steps,
        unit=" batches",
        leave=False,
        dynamic_ncols=True,
    )
    metrics = defaultdict(list)
    for i, batch in enumerate(pbar):
        rng_step, rng = random.split(rng)
        if i >= num_steps:  # for infinite dataloaders
            break
        m = valid_step(rng_step, state, batch)
        for k, v in m.items():
            metrics[k] += [v]
    return {k: np.mean(v) for k, v in metrics.items()}


def select_steps(model, is_categorical=False):
    train_step, valid_step = vanilla_train_step, vanilla_valid_step
    if isinstance(model, (NP, ANP)):
        train_step = npf_elbo_train_step
    elif isinstance(model, (BNP, BANP)):
        train_step, valid_step = bootstrap_train_step, bootstrap_valid_step
    elif isinstance(model, (TNPND,)):
        train_step, valid_step = tril_cov_train_step, tril_cov_valid_step
    if is_categorical:
        if isinstance(model, (BNP, BANP, TNPND)):
            raise NotImplementedError("Model does not support categorical loss!")
        train_step = partial(train_step, is_categorical=True)
        valid_step = partial(valid_step, is_categorical=True)
    return train_step, valid_step


@partial(jit, static_argnames=("is_categorical",))
def vanilla_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    is_categorical: bool = False,
    **kwargs,
):
    """Training step for meta regression with diagonal covariances.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.
        is_categorical: Indicates whether the output is categorical.

    Returns:
        `TrainState` with updated parameters.
    """
    rng_dropout, rng_extra = random.split(rng)

    def loss_fn(params):
        s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, *_ = batch
        (B, L_test, _) = s_test.shape
        if valid_lens_test is None:
            valid_lens_test = jnp.repeat(L_test, B)
        mask_test = mask_from_valid_lens(L_test, valid_lens_test)
        output = state.apply_fn(
            {"params": params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            valid_lens_test,
            training=True,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
        )
        if is_categorical:
            nll = safe_softmax_cross_entropy(output, f_test)
            return nll.mean(where=mask_test.squeeze())
        f_mu, f_std = output
        return -norm.logpdf(f_test, f_mu, f_std).mean(where=mask_test)

    nll, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), nll


@partial(jit, static_argnames=("is_categorical",))
def vanilla_valid_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    is_categorical: bool = False,
    **kwargs,
):
    s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, *_ = batch
    output = jit(state.apply_fn)(
        {"params": state.params, **state.kwargs},
        s_ctx,
        f_ctx,
        s_test,
        valid_lens_ctx,
        valid_lens_test,
        rngs={"extra": rng},
    )
    if isinstance(output[1], tuple):  # latent model
        output, _ = output  # latent zs aren't used in validation
    B, L_test, _ = s_test.shape
    if valid_lens_test is None:
        valid_lens_test = jnp.repeat(L_test, B)
    mask_test = mask_from_valid_lens(L_test, valid_lens_test)
    if is_categorical:
        nll = safe_softmax_cross_entropy(output, f_test)
        nll = nll.mean(where=mask_test.squeeze())
        return {"NLL": nll}
    hdi_prob = kwargs.get("hdi_prob", 0.95)
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    f_mu, f_std = output
    nll = -norm.logpdf(f_test, f_mu, f_std).mean(where=mask_test)
    rmse = jnp.sqrt(jnp.square(f_test - f_mu).mean(where=mask_test))
    mae = jnp.abs(f_test - f_mu).mean(where=mask_test)
    f_lower, f_upper = f_mu - z_score * f_std, f_mu + z_score * f_std
    cvg = ((f_test >= f_lower) & (f_test <= f_upper)).mean(where=mask_test)
    return {"NLL": nll, "RMSE": rmse, "MAE": mae, "Coverage": cvg}


@partial(jit, static_argnames=("is_categorical",))
def npf_elbo_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    is_categorical: bool = False,
    **kwargs,
):
    """Training step for meta regression with latents and diagonal covariances.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.
        is_categorical: Indicates whether the output is categorical.

    Returns:
        `TrainState` with updated parameters.
    """
    rng_dropout, rng_extra = random.split(rng)

    def loss_fn(params):
        s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, *_ = batch
        (B, L_test, _) = s_test.shape
        if valid_lens_test is None:
            valid_lens_test = jnp.repeat(L_test, B)
        mask_test = mask_from_valid_lens(L_test, valid_lens_test)
        output, (z_mu_ctx, z_std_ctx) = state.apply_fn(
            {"params": params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            valid_lens_test,
            training=True,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
        )
        # used by NP family only during training for KL-div loss term
        _, (z_mu_test, z_std_test) = state.apply_fn(
            {"params": params, **state.kwargs},
            s_test,
            f_test,
            s_test,
            valid_lens_test,
            valid_lens_test,
            training=True,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
        )
        # KL divergence and NLL assume diagonal covariance, i.e. pointwise.
        # Wikipedia's formulas for MVN KL-div: https://tinyurl.com/wiki-kl-div
        # Tensorflow's diagonal MVN KL-div impl (used here): https://tinyurl.com/diag-kl-div
        # KL( z_dist_test (p) || z_dist_ctx (q) ) =
        diff_log_scale = jnp.log(z_std_test) - jnp.log(z_std_ctx)
        kl_div = (
            0.5 * ((z_mu_test - z_mu_ctx) / z_std_ctx) ** 2
            + 0.5 * jnp.expm1(2 * diff_log_scale)
            - diff_log_scale
        ).sum(axis=-1) / valid_lens_test  # [B] <- avg KL per valid test loc
        if is_categorical:
            nll = safe_softmax_cross_entropy(output, f_test)
            nll = nll.mean(where=mask_test.squeeze())
        else:
            f_mu, f_std = output
            nll = -norm.logpdf(f_test, f_mu, f_std).mean(where=mask_test)
        return nll + kl_div.mean()

    elbo, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), elbo


@partial(jit, static_argnames=("is_categorical",))
def bootstrap_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    is_categorical: bool = False,
    **kwargs,
):
    """Training step for meta regression with diagonal covariances.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.
        is_categorical: Indicates whether the output is categorical.

    Returns:
        `TrainState` with updated parameters.
    """
    rng_dropout, rng_extra = random.split(rng)

    def loss_fn(params):
        s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, *_ = batch
        (B, L_test, _) = s_test.shape
        if valid_lens_test is None:
            valid_lens_test = jnp.repeat(L_test, B)
        mask_test = mask_from_valid_lens(L_test, valid_lens_test)
        output_boot, output = state.apply_fn(
            {"params": params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            valid_lens_test,
            training=True,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
        )
        if is_categorical:
            K = output_boot.shape[0] // B
            mask_test = mask_test.squeeze()
            f_test_boot = jnp.repeat(f_test, K, axis=0)
            nll_boot = safe_softmax_cross_entropy(output_boot, f_test_boot)
            nll_boot = logsumexp(nll_boot.reshape(B, K, L_test, -1)) - jnp.log(K)
            nll_boot = nll_boot.mean(where=mask_test)
            nll = safe_softmax_cross_entropy(output, f_test).mean(where=mask_test)
            # take average so it is on the scale as other train_step losses
            return (nll_boot + nll) / 2
        (f_mu_boot, f_std_boot), (f_mu, f_std) = output_boot, output
        K = f_mu_boot.shape[0] // B
        f_test_boot = jnp.repeat(f_test, K, axis=0)
        ll_boot = norm.logpdf(f_test_boot, f_mu_boot, f_std_boot)
        # log of likelihood averaged over K bootstrapped samples
        ll_boot = logsumexp(ll_boot.reshape(B, K, L_test, -1), axis=1) - jnp.log(K)
        nll_boot = -ll_boot.mean(where=mask_test)
        nll = -norm.logpdf(f_test, f_mu, f_std).mean(where=mask_test)
        # take average so it is on the scale as other train_step losses
        return (nll_boot + nll) / 2

    nll, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), nll


@partial(jit, static_argnames=("is_categorical",))
def bootstrap_valid_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    is_categorical: bool = False,
    **kwargs,
):
    s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, *_ = batch
    # at inference/validation time, only bootstrapped output is used
    # at training time, base output (ignored here) is also used
    output_boot, _ = jit(state.apply_fn)(
        {"params": state.params, **state.kwargs},
        s_ctx,
        f_ctx,
        s_test,
        valid_lens_ctx,
        valid_lens_test,
        rngs={"extra": rng},
    )
    B, L_test, _ = s_test.shape
    if valid_lens_test is None:
        valid_lens_test = jnp.repeat(L_test, B)
    mask_test = mask_from_valid_lens(L_test, valid_lens_test)
    if is_categorical:
        K = output_boot.shape[0] // B
        f_test_boot = jnp.repeat(f_test, K, axis=0)
        nll_boot = safe_softmax_cross_entropy(output_boot, f_test_boot)
        nll_boot = logsumexp(nll_boot.reshape(B, K, L_test, -1)) - jnp.log(K)
        nll_boot = nll_boot.mean(where=mask_test.squeeze())
        return {"NLL": nll_boot}
    hdi_prob = kwargs.get("hdi_prob", 0.95)
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    f_mu_boot, f_std_boot = output_boot
    K = f_mu_boot.shape[0] // B
    f_test_boot = jnp.repeat(f_test, K, axis=0)
    ll_boot = norm.logpdf(f_test_boot, f_mu_boot, f_std_boot)
    # log of likelihood averaged over K bootstrapped samples
    # NOTE: this is how BNP averages log-likelihood
    ll_boot = logsumexp(ll_boot.reshape(B, K, L_test, -1), axis=1) - jnp.log(K)
    nll = -ll_boot.mean(where=mask_test)
    rmse = jnp.sqrt(
        jnp.square(f_test_boot - f_mu_boot)
        .reshape(B, K, L_test, -1)
        .mean(axis=2, where=mask_test[:, None, ...])
    ).mean()  # average over [B, K]
    mae = (
        jnp.abs(f_test_boot - f_mu_boot)
        .reshape(B, K, L_test, -1)
        .mean(axis=2, where=mask_test[:, None, ...])
    ).mean()  # average over [B, K]
    f_lower = f_mu_boot - z_score * f_std_boot
    f_upper = f_mu_boot + z_score * f_std_boot
    cvg = (
        ((f_test_boot >= f_lower) & (f_test_boot <= f_upper))
        .reshape(B, K, L_test, -1)
        .mean(axis=2, where=mask_test[:, None, ...])
    ).mean()  # average over [B, K]
    return {"NLL": nll, "RMSE": rmse, "MAE": mae, "Coverage": cvg}


@jit
def tril_cov_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    **kwargs,
):
    """Training step for meta regression with lower triangular covariances, i.e.
    the L in a Cholesky decomposition.

    .. warning::
        This loss function ignores `valid_lens_test`, since jax doesn't provide
        a `where` mask argument in the multvariate normal's logpdf function.
        This means that the loss returned is taken over all test points for
        every item in the batch. This will produce incorrect results if the
        number of valid test points varies by item in the batch.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.

    Returns:
        `TrainState` with updated parameters.
    """
    rng_dropout, rng_extra = random.split(rng)

    def loss_fn(params):
        s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, *_ = batch
        f_mu, f_L = state.apply_fn(
            {"params": params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            valid_lens_test,
            training=True,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
        )
        B = f_test.shape[0]
        f_test_flat, f_mu_flat = f_test.reshape(B, -1), f_mu.reshape(B, -1)
        nlls = -mvn_logpdf(f_test_flat, f_mu_flat, f_L, is_tril=True)
        # average over valid lens to create average pointwise log-likelihood
        return (nlls / valid_lens_test).mean()

    nll, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), nll


@jit
def tril_cov_valid_step(rng: jax.Array, state: TrainState, batch: tuple, **kwargs):
    hdi_prob = kwargs.get("hdi_prob", 0.95)
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, *_ = batch
    f_mu, f_L, *_ = jit(state.apply_fn)(
        {"params": state.params, **state.kwargs},
        s_ctx,
        f_ctx,
        s_test,
        valid_lens_ctx,
        valid_lens_test,
        rngs={"extra": rng},
    )
    B = f_test.shape[0]
    mask_test = mask_from_valid_lens(s_test.shape[1], valid_lens_test)
    f_test_flat, f_mu_flat = f_test.reshape(B, -1), f_mu.reshape(B, -1)
    # WARNING: NLL ignores `valid_lens_test` because mvn_logpdf does yet support
    # masks with `where`.
    nlls = -mvn_logpdf(f_test_flat, f_mu_flat, f_L, is_tril=True)
    # average over valid lens to create average pointwise log-likelihood
    nll = (nlls / valid_lens_test).mean()
    rmse = jnp.sqrt(jnp.square(f_test - f_mu).mean(where=mask_test))
    mae = jnp.abs(f_test - f_mu).mean(where=mask_test)
    f_L_diag = vmap(jnp.diag)(f_L)[..., None]  # get marginal f_L
    f_lower, f_upper = f_mu - z_score * f_L_diag, f_mu + z_score * f_L_diag
    cvg = ((f_test >= f_lower) & (f_test <= f_upper)).mean(where=mask_test)
    return {"NLL": nll, "RMSE": rmse, "MAE": mae, "Coverage": cvg}


@partial(jit, static_argnames=("is_categorical",))
def fast_attention_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    is_categorical: bool = False,
    redraw_random_features: bool = False,
    **kwargs,
):
    """Training step for meta regression with diagonal covariances.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.
        is_categorical: Indicates whether the output is categorical.
        rng_redraw_randoM_features: An optional PRNG key used for redrawing
            random features used in fast attention kernel.

    Returns:
        `TrainState` with updated parameters.
    """
    rng_dropout, rng_extra = random.split(rng)

    def loss_fn(params):
        s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, *_ = batch
        (B, L_test, _) = s_test.shape
        if valid_lens_test is None:
            valid_lens_test = jnp.repeat(L_test, B)
        mask_test = mask_from_valid_lens(L_test, valid_lens_test)
        output, updated_state = state.apply_fn(
            {"params": params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            valid_lens_test,
            training=True,
            redraw_random_features=redraw_random_features,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
            mutable=["projections"],
        )
        if is_categorical:
            nll = safe_softmax_cross_entropy(output, f_test)
            return nll.mean(where=mask_test.squeeze())
        f_mu, f_std = output
        return -norm.logpdf(f_test, f_mu, f_std).mean(where=mask_test)

    (nll, updated_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )
    return state.apply_gradients(grads=grads, kwargs=updated_state), nll


def cfg_to_run_name(cfg: DictConfig):
    name = cfg.model.cls
    if "TNPKR" in name:
        name = {"TNPKR": "TNP-KR", "ScanTNPKR": "Scan TNP-KR"}[name]
        prefix = "model.kwargs.blk.kwargs.attn."
        attn_cls = OmegaConf.select(cfg, prefix + "cls")
        if attn_cls == "MultiHeadAttention":
            attn_cls = OmegaConf.select(cfg, prefix + "kwargs.attn.cls")
        name += ": " + attn_cls
    if name == "TNPD":
        name = "TNP-D"
    if name == "TNPND":
        name = "TNP-ND"
    return name


def save_ckpt(state: TrainState, cfg: DictConfig, path: Path):
    "Save a checkpoint."
    shutil.rmtree(path, ignore_errors=True)
    ckptr = PyTreeCheckpointer()
    ckpt = {"state": state, "config": OmegaConf.to_container(cfg, resolve=True)}
    save_args = orbax_utils.save_args_from_target(ckpt)
    ckptr.save(path.absolute(), ckpt, save_args=save_args)


def load_ckpts(
    dir: Union[str, Path],
    only_regex: Union[str, re.Pattern] = r".*",
    exclude_regex: Union[str, re.Pattern] = "$^",
):
    """Loads all checkpoints in a given base dir."""
    ckpt = {}
    if isinstance(only_regex, str):
        only_regex = re.compile(only_regex, re.IGNORECASE)
    if isinstance(exclude_regex, str):
        exclude_regex = re.compile(exclude_regex, re.IGNORECASE)
    for p in Path(dir).glob("*.ckpt"):
        if only_regex.match(str(p)) and not exclude_regex.match(str(p)):
            state, tmp_cfg = load_ckpt(p)
            ckpt[cfg_to_run_name(tmp_cfg)] = {"state": state, "cfg": tmp_cfg}
    return ckpt


def load_ckpt(path: Union[str, Path]):
    "Load a checkpoint."
    if not isinstance(path, Path):
        path = Path(path)
    ckptr = PyTreeCheckpointer()
    try:
        ckpt = ckptr.restore(path.absolute())
    except ValueError:
        sharding_file = path / "_sharding"
        cpu = str(jax.devices("cpu")[0])
        sharding = sharding_file.read_text().replace("cuda:0", cpu)
        sharding_file_bak = sharding_file.replace(path / "_sharding.bak")
        sharding_file.write_text(sharding)
        ckpt = ckptr.restore(path.absolute())
    finally:
        # cleanup
        if "sharding_file_bak" in locals() and sharding_file_bak.exists():
            sharding_file_bak.replace(sharding_file)
    cfg = OmegaConf.create(ckpt["config"])
    model = instantiate(cfg.model)
    state = TrainState.create(
        apply_fn=model.apply,
        # TODO(danj): reload optimizer state
        tx=optax.yogi(cosine_annealing_lr()),
        params=ckpt["state"]["params"],
        kwargs=ckpt["state"]["kwargs"],
    )
    return state, cfg


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


def custom_cosine_annealing_lr(num_steps: int, peak_lr: float):
    """Create a 3-cycle cosine annealing schedule.

    There are two cosine schedules each consisting of a quarter of `num_steps`
    and then a third single cosine schedule consisting of half of `num_steps`.
    """
    q, r = num_steps // 4, num_steps % 4
    q_sched = optax.cosine_onecycle_schedule(q, peak_lr, pct_start=0.2)
    h_sched = optax.cosine_onecycle_schedule(2 * q + r, peak_lr, pct_start=0.2)
    boundaries = [0, q, 2 * q]
    return optax.join_schedules([q_sched, q_sched, h_sched], boundaries)


def sample(
    rng: jax.Array,
    state: TrainState,
    s_ctx: jax.Array,  # [L_ctx, D_S]
    f_ctx: jax.Array,  # [L_ctx, D_F]
    s_test: jax.Array,  # [L_test, D_S]
    batch_size: int = 32,
):
    @jit
    def apply(s_ctx, f_ctx, s_test, valid_lens_ctx, rng_extra):
        return state.apply_fn(
            {"params": state.params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            training=False,
            rngs={"extra": rng_extra},
        )

    B, L_ctx, L_test = batch_size, s_ctx.shape[0], s_test.shape[0]
    s_ctx = jnp.repeat(jnp.vstack([s_ctx, s_test])[None, ...], B, axis=0)
    f_ctx = jnp.repeat(jnp.pad(f_ctx, ((0, L_test), (0, 0)))[None, ...], B, axis=0)
    s_test = jnp.repeat(s_test[None, ...], B, axis=0)
    valid_lens_ctx = jnp.repeat(L_ctx, B)
    for i in range(L_test):
        rng_extra, rng_eps, rng = random.split(rng, 3)
        output = apply(s_ctx, f_ctx, s_test, valid_lens_ctx, rng_extra)
        if isinstance(output[1], tuple):  # latent or bootstrapped
            output, _ = output  # throw away latent / base samples
        f_mu, f_std = output
        f_mu_i, f_std_i = f_mu[:, i, :], f_std[:, i, :]
        f_test_i = f_mu_i + f_std_i * random.normal(rng_eps, f_std_i.shape)
        f_ctx = f_ctx.at[:, L_ctx + i, :].set(f_test_i)
        valid_lens_ctx += 1
    return s_ctx[:, L_ctx:, :], f_ctx[:, L_ctx:, :]  # only return test locations


def log_img_plots(
    step: int,
    rng_step: int,
    state: TrainState,
    batch: tuple,
    shape: tuple[int, int, int],
    num_plots: int = 16,
    cmap=mpl.colormaps.get_cmap("grey"),
    cmap_std=mpl.colormaps.get_cmap("Spectral_r"),
    norm=None,
    norm_std=None,
    remap_colors: Callable = lambda x: x,
    transform_model_output: Callable = lambda x: x,
):
    """Logs `num_plots` from the given batch."""
    rng_dropout, rng_extra = random.split(rng_step)
    (
        s_ctx,
        f_ctx,
        valid_lens_ctx,
        s_test,
        f_test,
        valid_lens_test,
        s_test_full,
        f_test_full,
        inv_permute_idx,
    ) = batch
    output = state.apply_fn(
        {"params": state.params, **state.kwargs},
        s_ctx,
        f_ctx,
        s_test_full,
        valid_lens_ctx,
        valid_lens_test=None,
        rngs={"dropout": rng_dropout, "extra": rng_extra},
    )
    if isinstance(output[1], tuple):  # latent or bootstrapped
        output, _ = output  # throw away latent / base samples
    f_mu, f_std = transform_model_output(output)
    paths = []
    for i in range(num_plots):
        inv_idx_i = inv_permute_idx
        if inv_permute_idx.ndim > 1:  # each row was permuted separately
            inv_idx_i = inv_permute_idx[i]
        v = valid_lens_ctx[i]
        f_ctx_i = f_ctx[i, :v, :]
        f_mu_i = f_mu[i]
        f_std_i = f_std[i]
        f_test_full_i = f_test_full[i]
        if f_mu.shape != f_test.shape:  # bootstrapped
            K = f_mu.shape[0] // f_test.shape[0]
            s = i * K
            f_mu_i = f_mu[s : s + K].mean(axis=0)
            # Law of total variance: V[Y] = V[E[Y|X]] + E[V[Y|X]]
            f_std_i = f_mu[s : s + K].std(axis=0) + f_std[s : s + K].mean(axis=0)
        fig = plot_img(
            i,
            shape,
            f_ctx_i,
            f_mu_i,
            f_std_i,
            f_test_full_i,
            inv_idx_i,
            cmap=cmap,
            cmap_std=cmap_std,
            norm=norm,
            norm_std=norm_std,
            remap_colors=remap_colors,
        )
        paths += [f"/tmp/{datetime.now().isoformat()} - sample {i}.png"]
        fig.savefig(paths[-1], dpi=125)
        plt.clf()
        plt.close(fig)
    wandb.log({f"Step {step}": [wandb.Image(p) for p in paths]})


def regression_to_rgb(f: jax.Array):
    return jnp.clip(f / 2 + 0.5, 0, 1)  # [-1, 1] => [0, 1]


def plot_img(
    id: int,
    shape: tuple[int, int, int],
    f_ctx: jax.Array,  # [L_ctx, 1]
    f_mu: jax.Array,  # [L, 1]
    f_std: jax.Array,  # [L, 1]
    f_test: jax.Array,  # [L, 1]
    inv_permute_idx: jax.Array,  # [L]
    axs: Optional[mpl.axes.Axes] = None,
    cmap=mpl.colormaps.get_cmap("grey"),
    cmap_std=mpl.colormaps.get_cmap("Spectral_r"),
    norm=None,
    norm_std=None,
    remap_colors: Callable = lambda x: x,
):
    """Plots a triptych of [task, uncertainty, pred, truth]."""
    task = f_ctx_to_img_task(shape, f_ctx, inv_permute_idx)
    task_pred = f_mu.reshape(shape).squeeze()  # [H, W] or [H, W, D]
    task_true = f_test.reshape(shape).squeeze()
    task_std = f_std.reshape(shape).squeeze()
    if shape[-1] > 1:
        task_std = task_std.mean(axis=-1)
    if axs is None:
        _, axs = plt.subplots(1, 4, figsize=(20, 5))
    # NOTE: cmap and norm are ignored when the data has 3 channels, which it
    # assumes are RGB values; transform_rgb is a hack to get around this
    task = remap_colors(task)
    task_pred = remap_colors(task_pred)
    task_true = remap_colors(task_true)
    axs[0].set_title("Task")
    axs[0].imshow(task, cmap=cmap, norm=norm, interpolation="none")
    axs[1].set_title("Uncertainty")
    axs[1].imshow(task_std, cmap=cmap_std, norm=norm_std, interpolation="none")
    axs[2].set_title("Prediction")
    axs[2].imshow(task_pred, cmap=cmap, norm=norm, interpolation="none")
    axs[3].set_title("Ground Truth")
    axs[3].imshow(task_true, cmap=cmap, norm=norm, interpolation="none")
    plt.tight_layout()
    return plt.gcf()


def f_ctx_to_img_task(
    shape: tuple[int, int, int],
    f_ctx: jax.Array,
    inv_permute_idx: jax.Array,
):
    H, W, D = shape
    L, L_ctx = H * W, f_ctx.shape[0]
    task = jnp.pad(f_ctx, ((0, L - L_ctx), (0, 0)))  # [L_ctx, 1] -> [L, 1]
    task = task.at[L_ctx:, :].set(jnp.nan)  # will use cmap "bad" color
    task = task[inv_permute_idx, :]  # permute back to original ordering
    return task.reshape(shape).squeeze()  # [H, W] or # [H, W, D]


def log_wandb_line(vec: jax.Array, title: str):
    wandb_key = title.lower().replace(" ", "_")
    tbl = wandb.Table(data=[[i, v] for i, v in enumerate(vec)], columns=["i", "v"])
    wandb.log({wandb_key: wandb.plot.line(tbl, "i", "v", title=title)})
