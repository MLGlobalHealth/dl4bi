import pickle
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from flax.core import FrozenDict
from flax.training import orbax_utils, train_state
from jax import jit, random
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
from omegaconf import DictConfig, OmegaConf
from orbax.checkpoint import PyTreeCheckpointer
from sps.gp import GP
from sps.kernels import matern_3_2, periodic, rbf
from sps.priors import Prior
from sps.utils import build_grid
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
from .tnp_d import TNPD
from .tnp_ds import TNPDS
from .tnp_kr import TNPKR
from .tnp_nd import TNPND


@flax.struct.dataclass
class TrainState(train_state.TrainState):
    # kwargs stores any extra information associated with training,
    # i.e. batch norm stats or fixed (random) projections
    kwargs: FrozenDict = FrozenDict({})


@dataclass
class Callback:
    fn: Callable  # (step, rng_step, state, batch) -> None
    interval: int  # apply every interval of train_num_steps


def train(
    rng: jax.Array,
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    train_dataloader: Callable,
    valid_dataloader: Callable,
    train_num_steps: int = 100000,
    valid_num_steps: Optional[int] = None,
    valid_interval: int = 25000,
    log_loss_interval: int = 100,
    callbacks: list[Callback] = [],
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
        params=params,
        tx=optimizer,
        kwargs=kwargs,
    )
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
            valid_loss = evaluate(rng_valid, state, valid_dataloader, valid_num_steps)
            wandb.log({"valid_loss": valid_loss})
        for cbk in callbacks:
            if i % cbk.interval == 0:
                cbk.fn(i, rng_step, state, batch)
        pbar.set_postfix(
            {"train_loss": f"{train_loss:.3f}", "valid_loss": f"{valid_loss:.3f}"}
        )
    return state


def evaluate(
    rng: jax.Array,
    state: TrainState,
    dataloader: Callable,
    num_steps: Optional[float] = None,
    results_path: Optional[Path] = None,
):
    rng_data, rng_extra, rng_plots = random.split(rng, 3)
    losses, results = [], []
    num_steps = num_steps or float("inf")
    pbar = tqdm(
        dataloader(rng_data),
        total=num_steps,
        unit=" batches",
        leave=False,
        dynamic_ncols=True,
    )
    for i, batch in enumerate(pbar):
        # early stopping for infinite dataloaders
        if i >= num_steps:
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


def cfg_to_run_name(cfg: DictConfig):
    name = cfg.model.cls
    if name == "TNPKR":
        attn_cls = OmegaConf.select(
            cfg, "model.kwargs.dec.kwargs.blk.kwargs.attn.kwargs.attn.cls"
        )
        if attn_cls == "Attention":
            name += " Full"
        else:
            name += " " + attn_cls.replace("Attention", "")  # i.e. Fused, Fast
        embed_cls = OmegaConf.select(cfg, "model.kwargs.embed_s.cls")
        if embed_cls == "ResidualEmbedding":
            name += " Resid"
            embed_cls = OmegaConf.select(cfg, "model.kwargs.embed_s.kwargs.embed.cls")
        if embed_cls in [
            "GaussianFourierEmbedding",
            "NeRFEmbedding",
            "FixedSinusoidalEmbedding",
        ]:
            name += " RFF"
    return name


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


def build_gp_dataloader(data: DictConfig, kernel: DictConfig):
    """Generates batches of GP samples."""
    gp = instantiate(kernel)
    s_dim = len(data.s)
    s_grid = build_grid(data.s).reshape(-1, s_dim)  # flatten spatial dims
    obs_noise, batch_size = data.obs_noise, data.batch_size
    valid_lens_test = jnp.repeat(data.num_ctx.max + s_grid.shape[0], batch_size)
    min_s = jnp.array([axis["start"] for axis in data.s])
    max_s = jnp.array([axis["stop"] for axis in data.s])

    @jit
    def gen_s_random(rng: jax.Array):
        return random.uniform(
            rng, (data.num_ctx.max, s_dim), minval=min_s, maxval=max_s
        )

    def gen_batch(rng: jax.Array):
        rng_s_random, rng_valid_lens_ctx, rng_gp, rng_eps, rng = random.split(rng, 5)
        s_random = gen_s_random(rng_s_random)
        s = jnp.vstack([s_random, s_grid])
        f, var, ls, period, *_ = gp.simulate(rng_gp, s, batch_size)
        valid_lens_ctx = random.randint(
            rng_valid_lens_ctx,
            (batch_size,),
            data.num_ctx.min,
            data.num_ctx.max,
        )
        s = jnp.repeat(s[None, ...], batch_size, axis=0)
        f_noisy = f + obs_noise * random.normal(rng_eps, f.shape)
        return s, f_noisy, valid_lens_ctx, s, f, valid_lens_test, var, ls, period

    def dataloader(rng: jax.Array):
        while True:
            rng_batch, rng = random.split(rng)
            yield gen_batch(rng_batch)

    return dataloader


def save_ckpt(state: TrainState, cfg: DictConfig, path: Path):
    "Save a checkpoint."
    shutil.rmtree(path, ignore_errors=True)
    ckptr = PyTreeCheckpointer()
    ckpt = {"state": state, "config": OmegaConf.to_container(cfg, resolve=True)}
    save_args = orbax_utils.save_args_from_target(ckpt)
    ckptr.save(path.absolute(), ckpt, save_args=save_args)


def load_ckpt(path: Path):
    "Load a checkpoint."
    ckptr = PyTreeCheckpointer()
    ckpt = ckptr.restore(path.absolute())
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
def fast_attention_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    redraw_random_features: bool = False,
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
            redraw_random_features=redraw_random_features,
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
        f_mu, f_std, *_ = apply(s_ctx, f_ctx, s_test, valid_lens_ctx, rng_extra)
        f_mu_i, f_std_i = f_mu[:, i, :], f_std[:, i, :]
        f_test_i = f_mu_i + f_std_i * random.normal(rng_eps, f_std_i.shape)
        f_ctx = f_ctx.at[:, L_ctx + i, :].set(f_test_i)
        valid_lens_ctx += 1
    return s_ctx[:, L_ctx:, :], f_ctx[:, L_ctx:, :]  # only return test locations


def plot_posterior_predictives(
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    valid_lens_ctx: jax.Array,
    s_test: jax.Array,
    f_test: jax.Array,
    valid_lens_test: jax.Array,
    var: jax.Array,
    ls: jax.Array,
    period: jax.Array | None,
    f_mu: jax.Array,
    f_std: jax.Array,
    num_plots: int = 16,
):
    """Plots `num_plots` from the given batch."""
    paths = []
    for i in range(num_plots):
        v_ctx = valid_lens_ctx[i]
        s_ctx_i = s_ctx[i, :v_ctx].squeeze()
        f_ctx_i = f_ctx[i, :v_ctx].squeeze()
        v_test = valid_lens_test[i]
        s_test_i = s_test[i, :v_test].squeeze()
        f_test_i = f_test[i, :v_test].squeeze()
        f_mu_i = f_mu[i, :v_test].squeeze()
        f_std_i = f_std[i, :v_test].squeeze()
        if f_mu[i].shape != f_std[i].shape:  # marginal from tril cov
            f_std_i = jnp.diag(f_std[i]).squeeze()  # TODO(danj): is this valid?
        if f_mu.shape != f_test.shape:  # bootstrapped
            K = f_mu.shape[0] // f_test.shape[0]
            s = i * K
            f_mu_i = f_mu[s : s + K].squeeze()
            f_std_i = f_std[s : s + K].squeeze()
        title = f"Sample {i} (var: {var[i]:0.2f}, ls: {ls[i]:0.2f}"
        title += f", period: {period[i]:0.2f})" if jnp.isfinite(period) else ")"
        fig = plot_posterior_predictive(
            s_ctx_i, f_ctx_i, s_test_i, f_test_i, f_mu_i, f_std_i
        )
        fig.suptitle(title)
        paths += [f"/tmp/{datetime.now().isoformat()} - {title}.png"]
        fig.savefig(paths[-1], dpi=125)
        plt.clf()
        plt.close(fig)
    return paths


def plot_posterior_predictive(
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    s_test: jax.Array,
    f_test: jax.Array,
    f_mu: jax.Array,
    f_std: jax.Array,
    hdi_prob: float = 0.95,
):
    """Plots the posterior predictive alongside true values."""
    f_mu = f_mu[None, ...] if f_mu.ndim == 1 else f_mu
    f_std = f_std[None, ...] if f_std.ndim == 1 else f_std
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    f_lower, f_upper = f_mu - z_score * f_std, f_mu + z_score * f_std
    idx = jnp.argsort(s_test)
    plt.plot(s_test[idx], f_test[idx], color="black")
    plt.scatter(s_ctx, f_ctx, color="black", alpha=0.75)
    K = f_mu.shape[0]
    for i in range(K):
        f_mu_i = f_mu[i]
        plt.plot(s_test[idx], f_mu_i[idx], color="steelblue")
        f_lower_i, f_upper_i = f_lower[i], f_upper[i]
        plt.fill_between(
            s_test[idx],
            f_lower_i[idx],
            f_upper_i[idx],
            alpha=0.4 / K,
            color="steelblue",
            interpolate=True,
        )
    ax = plt.gca()
    ax.set_xlabel("s")
    ax.set_ylabel("f")
    return plt.gcf()


def log_posterior_predictive_plots(
    step: int,
    rng_step: int,
    state: TrainState,
    batch: tuple,
    num_plots: int = 16,
):
    rng_dropout, rng_extra = random.split(rng_step)
    s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, var, ls, period = (
        batch
    )
    f_mu, f_std, *_ = state.apply_fn(
        {"params": state.params, **state.kwargs},
        s_ctx,
        f_ctx,
        s_test,
        valid_lens_ctx,
        valid_lens_test,
        rngs={"dropout": rng_dropout, "extra": rng_extra},
    )
    paths = plot_posterior_predictives(
        s_ctx,
        f_ctx,
        valid_lens_ctx,
        s_test,
        f_test,
        valid_lens_test,
        var,
        ls,
        period,
        f_mu,
        f_std,
        num_plots,
    )
    wandb.log({f"Step {step}": [wandb.Image(p) for p in paths]})


def log_img_plots(
    step: int,
    rng_step: int,
    state: TrainState,
    batch: tuple,
    shape: tuple[int, int, int],
    num_plots: int = 16,
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
    f_mu, f_std, *_ = state.apply_fn(
        {"params": state.params, **state.kwargs},
        s_ctx,
        f_ctx,
        s_test_full,
        valid_lens_ctx,
        valid_lens_test=None,
        rngs={"dropout": rng_dropout, "extra": rng_extra},
    )
    paths = []
    for i in range(num_plots):
        v = valid_lens_ctx[i]
        f_ctx_i = f_ctx[i, :v, :]
        f_mu_i = f_mu[i]
        f_test_full_i = f_test_full[i]
        if f_mu.shape != f_test.shape:  # bootstrapped
            K = f_mu.shape[0] // f_test.shape[0]
            s = i * K
            f_mu_i = f_mu[s : s + K].mean(axis=0)  # TODO(danj): legitimate?
        fig = plot_img(i, shape, f_ctx_i, f_mu_i, f_test_full_i, inv_permute_idx)
        paths += [f"/tmp/sample_{i}.png"]
        fig.savefig(paths[-1], dpi=125)
        plt.clf()
        plt.close(fig)
    wandb.log({f"Step {step}": [wandb.Image(p) for p in paths]})


def plot_img(
    id: int,
    shape: tuple[int, int, int],
    f_ctx: jax.Array,  # [L_ctx, 1]
    f_mu: jax.Array,  # [L, 1]
    f_test: jax.Array,  # [L, 1]
    inv_permute_idx: jax.Array,  # [L]
):
    """Plots a triptych of [task, pred, truth]."""
    task = f_ctx_to_img_task(shape, f_ctx, inv_permute_idx)
    task_pred = f_to_img_task(shape, f_mu)
    task_true = f_to_img_task(shape, f_test)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(task)
    axs[0].set_title("Task")
    axs[1].imshow(task_pred)
    axs[1].set_title("Predicted")
    axs[2].imshow(task_true)
    axs[2].set_title("Ground Truth")
    plt.tight_layout()
    return plt.gcf()


def f_ctx_to_img_task(
    shape: tuple[int, int, int],
    f_ctx: jax.Array,
    inv_permute_idx: jax.Array,
):
    H, W, D = shape
    L, L_ctx = H * W, f_ctx.shape[0]
    f_ctx = f_ctx / 2 + 0.5  # [-1, 1] -> [0, 1]
    task = jnp.pad(f_ctx, ((0, L - L_ctx), (0, 0)))  # [L_ctx, 1] -> [L, 1]
    if D == 1:  # if black/white, convert to RGB
        task = jnp.repeat(task, 3, axis=-1)  # [L, 1] -> [L, 3]
    task = task.at[L_ctx:, 2].set(1.0)  # set non-context points to blue
    task = task[inv_permute_idx, :]  # permute back to original ordering
    return task.reshape(*shape[:-1], 3)  # reshape to [H, W, 3] image


def f_to_img_task(shape: tuple[int, int, int], f: jax.Array):
    task = f.reshape(shape)  # [H, W, D]
    task = task / 2 + 0.5  # [-1, 1] -> [0, 1]
    if shape[-1] == 1:  # if black/white, convert to RGB
        task = jnp.repeat(task, 3, axis=-1)  # [H, W, 3]
    return jnp.clip(task, 0, 1)  # to avoid matplotlib warnings


def log_wandb_line(vec: jax.Array, title: str):
    wandb_key = title.lower().replace(" ", "_")
    tbl = wandb.Table(data=[[i, v] for i, v in enumerate(vec)], columns=["i", "v"])
    wandb.log({wandb_key: wandb.plot.line(tbl, "i", "v", title=title)})
