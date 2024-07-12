import pickle
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from flax.core import FrozenDict
from flax.training import train_state
from jax import jit, random
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
from omegaconf import DictConfig, OmegaConf
from sps.gp import GP
from sps.kernels import matern_3_2, periodic, rbf
from sps.priors import Prior
from tqdm import tqdm

from ..core import *
from .anp import ANP
from .banp import BANP
from .bnp import BNP
from .canp import CANP
from .cnp import CNP
from .convcnp import ConvCNP
from .dkr import DKR
from .np import NP
from .sptx import SPTx
from .tnpd import TNPD
from .tnpds import TNPDS
from .tnpnd import TNPND


class TrainState(train_state.TrainState):
    kwargs: FrozenDict = FrozenDict({})


@dataclass
class Callback:
    fn: Callable  # (step, rng_step, state, batch) -> None
    interval: int  # apply every interval of train_num_steps


def train(
    rng: jax.Array,
    model_cfg: DictConfig,
    optimizer: optax.GradientTransformation,
    train_dataloader: Callable,
    valid_dataloader: Callable,
    train_num_steps: int = 100000,
    valid_num_steps: Optional[int] = None,
    valid_interval: int = 25000,
    log_loss_interval: int = 100,
    lr_peak: float = 1e-3,
    lr_pct_warmup: float = 0.3,
    lr_num_cycles: int = 1,
    callbacks: list[Callback] = [],
):
    model = instantiate(model_cfg)
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
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        kwargs=kwargs,
    )
    print(f"{model}\n\n{param_count}")
    train_step = vanilla_train_step
    if isinstance(model, (NP, ANP)):
        train_step = npf_elbo_train_step
    elif isinstance(model, (BNP, BANP)):
        train_step = bootstrap_train_step
    elif isinstance(model, (TNPND,)):
        train_step = tril_cov_train_step
    losses = []
    train_loss, valid_loss = float("inf"), float("inf")
    pbar = tqdm(range(1, train_num_steps + 1), unit=" batches", dynamic_ncols=True)
    for i in pbar:
        batch = next(batches)
        rng_step, rng_train = random.split(rng_train)
        state, loss = train_step(rng_step, state, batch)
        losses += [loss]
        if i % log_loss_interval == 0:
            train_loss = np.mean(losses[i - log_loss_interval : i])
            wandb.log({"train_loss": train_loss})
        if i % valid_interval == 0:
            rng_valid, rng_train = random.split(rng_train)
            valid_loss = validate(rng_valid, state, valid_dataloader, valid_num_steps)
            wandb.log({"valid_loss": valid_loss})
        for cbk in callbacks:
            if i % cbk.interval == 0:
                cbk.fn(i, rng_step, state, batch)
        pbar.set_postfix(
            {"train_loss": f"{train_loss:.3f}", "valid_loss": f"{valid_loss:.3f}"}
        )
    return state


def validate(
    rng: jax.Array,
    state: TrainState,
    valid_dataloader: Callable,
    valid_num_steps: Optional[float] = None,
    results_path: Optional[Path] = None,
):
    rng_data, rng_extra, rng_plots = random.split(rng, 3)
    losses, results = [], []
    valid_num_steps = valid_num_steps or float("inf")
    pbar = tqdm(
        valid_dataloader(rng_data),
        total=valid_num_steps,
        unit=" batches",
        leave=False,
        dynamic_ncols=True,
    )
    for i, batch in enumerate(pbar):
        # early stopping for infinite dataloaders
        if i >= valid_num_steps:
            break
        s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, *_ = batch
        f_mu, f_std, *_ = jit(state.apply_fn)(
            {"params": state.params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            valid_lens_test,
            rngs={"extra": rng_extra},
        )
        if f_mu.shape == f_std.shape:  # f_std is independent/diagonal
            mask_test = mask_from_valid_lens(s_test.shape[1], valid_lens_test)
            if f_mu.shape == f_ctx.shape:
                losses += [-norm.logpdf(f_test, f_mu, f_std).mean(where=mask_test)]
            else:  # bootstrapped
                B, L_test, _ = s_test.shape
                K = f_mu.shape[0] // f_ctx.shape[0]
                f_test_boot = jnp.repeat(f_test, K, axis=0)
                ll_boot = norm.logpdf(f_test_boot, f_mu, f_std)
                # log of likelihood averaged over K bootstrapped samples
                ll_boot = logsumexp(
                    ll_boot.reshape(B, K, L_test, -1), axis=1
                ) - jnp.log(K)
                losses += [-ll_boot.mean(where=mask_test)]
        else:  # f_std is a lower triangular covariance matrix, e.g. TNPND
            # WARNING: This ignores `valid_lens_test` because
            # mvn_logpdf does yet support masks with `where`.
            B = f_test.shape[0]
            f_test_flat, f_mu_flat = f_test.reshape(B, -1), f_mu.reshape(B, -1)
            nlls = -mvn_logpdf(f_test_flat, f_mu_flat, f_std, is_tril=True)
            # average over valid lens to create average pointwise log-likelihood
            losses += [(nlls / valid_lens_test).mean()]
        if results_path:
            b = [np.array(v) for v in batch]
            p = [np.array(v) for v in [f_mu, f_std]]
            results += [(b, p)]
    loss = np.mean(losses)
    if results_path:
        with open(results_path, "wb") as f:
            pickle.dump(results, f)
    return loss


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


def save_ckpt(state: TrainState, cfg: DictConfig, path: Path):
    """Saves a checkpoint."""
    shutil.rmtree(path, ignore_errors=True)
    ckptr = ocp.Checkpointer(ocp.CompositeCheckpointHandler("state", "config"))
    cfg_d = OmegaConf.to_container(cfg, resolve=True)
    ckptr.save(
        path.absolute(),
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(state),
            config=ocp.args.JsonSave(cfg_d),
        ),
    )


def load_ckpt(path: Path, sample_batch: tuple):
    """Loads a checkpoint."""
    ckptr = ocp.Checkpointer(ocp.CompositeCheckpointHandler("state", "config"))
    # restore config and use it to create model template
    ckpt = ckptr.restore(path, args=ocp.args.Composite(config=ocp.args.JsonRestore()))
    cfg = OmegaConf.create(ckpt["config"])
    rng, num_steps = random.key(42), 100000
    lr_peak, lr_pct_warmup, lr_num_cycles = 1e-3, 0.1, 1
    s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, *_ = sample_batch
    model = instantiate(cfg.model)
    kwargs = model.init(rng, s_ctx, f_ctx, s_test, valid_lens_ctx, valid_lens_test)
    params = kwargs.pop("params")
    learning_rate_fn = create_learning_rate_fn(
        num_steps,
        lr_peak,
        lr_pct_warmup,
        lr_num_cycles,
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.yogi(learning_rate_fn),
        kwargs=kwargs,
    )
    ckpt = ckptr.restore(
        path, args=ocp.args.Composite(state=ocp.args.StandardRestore(state))
    )
    return ckpt["state"], cfg


def cosine_annealing_lr(
    num_steps: int = 100000,
    peak_lr: float = 1e-3,
    pct_warmup: float = 0.3,
    num_cycles: int = 1,
):
    """Create an n-cycle cosine annealing schedule."""
    n = num_steps // num_cycles
    sched = optax.cosine_onecycle_schedule(n, peak_lr, pct_start=pct_warmup)
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


@jit
def vanilla_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    **kwargs,
):
    """Training step for meta regression with diagonal covariances.

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
        (B, L_test, _) = s_test.shape
        if valid_lens_test is None:
            valid_lens_test = jnp.repeat(L_test, B)
        mask_test = mask_from_valid_lens(L_test, valid_lens_test)
        f_mu, f_std = state.apply_fn(
            {"params": params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            valid_lens_test,
            training=True,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
        )
        return -norm.logpdf(f_test, f_mu, f_std).mean(where=mask_test)

    nll, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), nll


@jit
def bootstrap_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    **kwargs,
):
    """Training step for meta regression with diagonal covariances.

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
        (B, L_test, _) = s_test.shape
        if valid_lens_test is None:
            valid_lens_test = jnp.repeat(L_test, B)
        mask_test = mask_from_valid_lens(L_test, valid_lens_test)
        f_mu_boot, f_std_boot, f_mu, f_std = state.apply_fn(
            {"params": params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            valid_lens_test,
            training=True,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
        )
        K = f_mu_boot.shape[0] // f_mu.shape[0]
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
            rngs={"dropout": rng_dropout, "extra": rng_dropout},
        )
        B = f_test.shape[0]
        f_test_flat, f_mu_flat = f_test.reshape(B, -1), f_mu.reshape(B, -1)
        nlls = -mvn_logpdf(f_test_flat, f_mu_flat, f_L, is_tril=True)
        # average over valid lens to create average pointwise log-likelihood
        return (nlls / valid_lens_test).mean()

    nll, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), nll


@jit
def fast_attention_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    rng_redraw_random_features: Optional[jax.Array] = None,
    **kwargs,
):
    """Training step for meta regression with diagonal covariances.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.
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
        (f_mu, f_std), updated_state = state.apply_fn(
            {"params": params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            valid_lens_test,
            training=True,
            rng_redraw_random_features=rng_redraw_random_features,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
            mutable=["projections"],
        )
        return -norm.logpdf(f_test, f_mu, f_std).mean(where=mask_test), updated_state

    (nll, updated_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )
    return state.apply_gradients(grads=grads, kwargs=updated_state), nll


@jit
def npf_elbo_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    **kwargs,
):
    """Training step for meta regression with latents and diagonal covariances.

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
        (B, L_test, _) = s_test.shape
        if valid_lens_test is None:
            valid_lens_test = jnp.repeat(L_test, B)
        mask_test = mask_from_valid_lens(L_test, valid_lens_test)
        f_mu, f_std, z_mu_ctx, z_std_ctx = state.apply_fn(
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
        _, _, z_mu_test, z_std_test = state.apply_fn(
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
        nll = -norm.logpdf(f_test, f_mu, f_std).mean(where=mask_test)
        return nll + kl_div.mean()

    elbo, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), elbo
