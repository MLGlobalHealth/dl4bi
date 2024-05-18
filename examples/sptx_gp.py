#!/usr/bin/env python3
import pickle
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Union

import arviz as az
import flax
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
import orbax.checkpoint as ocp
from clu import metrics
from flax import struct
from flax.core import FrozenDict
from flax.training import train_state
from jax import Array, grad, jit, random
from jax.scipy.stats import norm
from jax.tree_util import Partial
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from omegaconf import DictConfig, OmegaConf
from sps.gp import GP
from sps.kernels import Kernel, matern_3_2, periodic, rbf
from sps.priors import Prior
from sps.utils import build_grid
from tqdm import tqdm

from dge import *


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics
    kwargs: FrozenDict = FrozenDict({})


@dataclass
class Task:
    name: str
    kernel: Kernel
    var: Prior
    ls: Prior


@hydra.main("configs", "sptx_gp", None)
def main(cfg: DictConfig):
    key = random.key(0)
    OmegaConf.register_new_resolver("eval", eval)
    s = build_grid(cfg.data.grid)
    periodic_0_1 = Partial(periodic, period=cfg.data.period)
    var, ls = Prior("fixed", {"value": 1.0}), Prior("beta", {"a": 2.5, "b": 6})
    periodic_task = Task(name="Periodic", kernel=periodic_0_1, var=var, ls=ls)
    matern_3_2_task = Task(name="Matern 3-2", kernel=matern_3_2, var=var, ls=ls)
    rbf_task = Task(name="RBF", kernel=rbf, var=var, ls=ls)
    # for task in [rbf_task, matern_3_2_task, periodic_task]:
    for task in [rbf_task]:  # matern_3_2_task, periodic_task]:
        print(task.name)
        rng_loader, rng_hmc, rng_tr, key = random.split(key, 4)
        gp = GP(task.kernel, task.var, task.ls)
        loader = dataloader(rng_loader, gp, s, **cfg.data.loader)
        (s_ctx, f_ctx, valid_lens), (s_test, f_test, f_noisy) = next(loader)
        state = train(cfg, loader, rng_tr)
        # save_ckpt(state, cfg)
        # state, cfg = load_ckpt(cfg.train.ckpt_path)
        valid_lens = valid_lens.at[0].set(cfg.data.num_test)
        f_dist = state.apply_fn(
            {"params": state.params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens,
        )
        f_mu, f_log_var = f_dist[..., [0]], f_dist[..., [1]]
        for i in range(s_ctx.shape[0]):
            f_mu_i, f_log_var_i = f_mu[i].squeeze(), f_log_var[i].squeeze()
            s_ctx_i, f_ctx_i = s_ctx[i].squeeze(), f_ctx[i].squeeze()
            s_test_i, f_test_i = s_test[i].squeeze(), f_test[i].squeeze()
            f_noisy_i, valid_len_i = f_noisy[i].squeeze(), valid_lens[i]
            name = f"{task.name} ({i})"
            plot_posterior_predictive_params(
                name,
                s_ctx_i,
                f_ctx_i,
                valid_len_i,
                s_test_i,
                f_test_i,
                f_noisy_i,
                f_mu_i,
                f_log_var_i,
            )
        # gp_model = build_gp_model(task.kernel)
        # pp = hmc(task, gp_model, rng_hmc, s_ctx, f_ctx, valid_len, cfg.infer)
        # plot_posterior_predictive_samples(
        #     task.name, s_ctx, f_ctx, valid_len, s_test, f_test, f_noisy, pp["obs"]
        # )


def dataloader(
    key,
    gp,
    s,
    obs_noise=0.1,
    min_p=0.05,
    max_p=0.5,
    batch_size=64,
    approx=False,
):
    S = s.shape[0]
    _s = jnp.repeat(s[None, ...], batch_size, axis=0)  # [B, S, D_S]
    min_obs, max_obs = int(min_p * S), int(max_p * S)

    @jit
    def gen_batch(rng: jax.Array):
        rng_gp, rng_noise, rng_perm, rng_valid = random.split(rng, 4)
        _var, _ls, _z, f = gp.simulate(rng_gp, s, batch_size, approx)
        valid_lens = random.randint(rng_valid, (batch_size,), min_obs, max_obs)
        perm = random.permutation(rng_perm, S)
        s_perm, f_perm = _s[:, perm, :], f[:, perm, :]
        f_perm_noisy = f_perm + obs_noise * random.normal(rng_noise, f.shape)
        return (s_perm, f_perm_noisy, valid_lens), (s_perm, f_perm, f_perm_noisy)

    while True:
        rng, key = random.split(key)
        yield gen_batch(rng)


def train(cfg: DictConfig, loader: Iterable, rng: Array):
    rng_model, rng_init, rng_train = random.split(rng, 3)
    model = instantiate(cfg.model)
    (s_ctx, f_ctx, valid_lens), (s_test, _, _) = next(loader)
    kwargs = model.init(rng_init, s_ctx, f_ctx, s_test, valid_lens)
    params = kwargs.pop("params")
    # learning_rate_fn = create_learning_rate_fn(
    #     cfg.train.num_batches,
    #     cfg.train.num_warmup,
    #     cfg.train.learning_rate,
    # )
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        # tx=optax.adamaxw(learning_rate_fn),
        tx=optax.yogi(cfg.train.learning_rate),
        metrics=Metrics.empty(),
        kwargs=kwargs,
    )
    metrics = {"train_loss": []}
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"{model}\n\nParam count: {param_count}")
    with tqdm(range(1, cfg.train.num_batches + 1), unit="batch") as pbar:
        rng_dropout, rng_train = random.split(rng_train)
        for i in pbar:
            batch = next(loader)
            rng_redraw_random_features = None
            if i % cfg.train.redraw_random_features_every_n == 0:
                rng_redraw_random_features, rng_train = random.split(rng_train)
            state = train_step(rng_dropout, state, batch, rng_redraw_random_features)
            if i % 10 == 0:
                state = compute_metrics(state, batch)
                for metric, value in state.metrics.compute().items():
                    metrics[f"train_{metric}"].append(value)
                state = state.replace(metrics=state.metrics.empty())
                pbar.set_postfix(nll=f"{metrics['train_loss'][-1]:.3f}")
    return state


def instantiate(d: Union[dict, DictConfig]):
    if isinstance(d, DictConfig):
        d = OmegaConf.to_container(d, resolve=True)
    for k in d:
        if isinstance(d[k], dict):
            d[k] = instantiate(d[k])
    if "cls" in d:
        cls, kwargs = d["cls"], d.get("kwargs", {})
        return globals()[cls](**kwargs)
    elif "func" in d:
        return eval(d["func"])
    return d


@jit
def train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    rng_redraw_random_features: Optional[jax.Array] = None,
):
    def loss_fn(params):
        (s_ctx, f_ctx, valid_lens), (s_test, f_test, f_test_noisy) = batch
        f_dist, updated_state = state.apply_fn(
            {"params": params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens,
            training=True,
            rng_redraw_random_features=rng_redraw_random_features,
            rngs={"dropout": rng},
            mutable=["projections"],
        )
        f_mu, f_log_var = f_dist[..., [0]], f_dist[..., [1]]
        nll = -norm.logpdf(f_test_noisy, f_mu, jnp.exp(f_log_var / 2)).mean()
        return nll, updated_state

    (nll, updated_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )
    return state.apply_gradients(grads=grads, kwargs=updated_state)


def create_learning_rate_fn(
    num_steps: int,
    num_warmup_steps: int,
    peak_learning_rate: float = 1e-3,
):
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=peak_learning_rate, transition_steps=num_warmup_steps
    )
    decay_steps = num_steps - num_warmup_steps
    cosine_fn = optax.cosine_decay_schedule(peak_learning_rate, decay_steps)
    schedule_fn = optax.join_schedules(
        [warmup_fn, cosine_fn], boundaries=[num_warmup_steps]
    )
    return schedule_fn


@jit
def compute_metrics(state, batch):
    (s_ctx, f_ctx, valid_lens), (s_test, f_test, f_test_noisy) = batch
    f_dist = state.apply_fn(
        {"params": state.params, **state.kwargs},
        s_ctx,
        f_ctx,
        s_test,
        valid_lens,
    )
    f_mu, f_log_var = f_dist[..., [0]], f_dist[..., [1]]
    nll = -norm.logpdf(f_test_noisy, f_mu, jnp.exp(f_log_var / 2)).mean()
    metric_updates = state.metrics.single_from_model_output(loss=nll)
    metrics = state.metrics.merge(metric_updates)
    return state.replace(metrics=metrics)


def save_ckpt(state: TrainState, cfg: DictConfig):
    path = Path(cfg.train.ckpt_path).absolute()
    shutil.rmtree(path, ignore_errors=True)
    ckptr = ocp.Checkpointer(ocp.CompositeCheckpointHandler("state", "config"))
    cfg_d = OmegaConf.to_container(cfg, resolve=True)
    ckptr.save(
        path,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(state),
            config=ocp.args.JsonSave(cfg_d),
        ),
    )


def load_ckpt(path: str):
    p = Path(path).absolute()
    ckptr = ocp.Checkpointer(ocp.CompositeCheckpointHandler("state", "config"))
    ckpt = ckptr.restore(p, args=ocp.args.Composite(config=ocp.args.JsonRestore()))
    cfg = OmegaConf.create(ckpt["config"])
    model = instantiate(cfg.model)
    B, L, D = 4, cfg.data.grid[0].num, 1
    s = f = jnp.zeros((B, L, D))
    valid_lens = jnp.repeat(L, B)
    key = random.key(42)
    kwargs = model.init(key, s, f, s, valid_lens, valid_lens)
    params = kwargs.pop("params")
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.yogi(1e-3),
        metrics=Metrics.empty(),
        kwargs=kwargs,
    )
    ckpt = ckptr.restore(
        p, args=ocp.args.Composite(state=ocp.args.StandardRestore(state))
    )
    return ckpt["state"], cfg


def plot_posterior_predictive_params(
    name,
    s_ctx,
    f_ctx,
    valid_len,
    s_test,
    f_test,
    f_test_noisy,
    f_mu,
    f_log_var,
    hdi_prob=0.9,
):
    idx = jnp.argsort(s_test)
    s_test, f_test = s_test[idx], f_test[idx]
    f_mu, f_log_var = f_mu[idx], f_log_var[idx]
    f_std = jnp.exp(f_log_var / 2)
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    f_lower, f_upper = f_mu - z_score * f_std, f_mu + z_score * f_std
    plt.plot(s_test, f_test, color="black")
    plt.plot(s_test, f_mu, color="steelblue")
    plt.scatter(s_ctx[:valid_len], f_ctx[:valid_len], color="black")
    plt.fill_between(
        s_test,
        f_lower,
        f_upper,
        alpha=0.4,
        color="steelblue",
        interpolate=True,
    )
    ax = plt.gca()
    ax.set_xlabel("s")
    ax.set_ylabel("f")
    plt.title(f"SPTx: {name} Posterior Predictive")
    plt.savefig(f"plots/SPTx: {name} Posterior Predictive.pdf", dpi=600)
    plt.clf()


def build_gp_model(kernel):
    def m(s_ctx, f_ctx=None, valid_len=None):
        variance = numpyro.sample("variance", dist.HalfNormal())
        lengthscale = numpyro.sample("lengthscale", dist.HalfNormal())
        # jitter added on diagonal for cholesky decomposition stability
        K = kernel(s_ctx, s_ctx, variance, lengthscale) + 1e-6 * jnp.eye(len(s_ctx))
        f_mu = numpyro.sample("f_mu", dist.MultivariateNormal(0, K))
        f_sigma = numpyro.sample("f_sigma", dist.HalfNormal(0.1))
        if valid_len:
            f_ctx, f_mu = (f_ctx[:valid_len], f_mu[:valid_len])
        numpyro.sample("obs", dist.Normal(f_mu, f_sigma), obs=f_ctx)

    return m


def hmc(task, model, rng, s_ctx, f_ctx, valid_len, cfg: DictConfig):
    rng_mcmc, rng_pp = random.split(rng)
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    mcmc = MCMC(nuts, **cfg)
    mcmc.run(rng_mcmc, s_ctx, f_ctx, valid_len)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    pp = Predictive(model, samples)(rng_pp, s_ctx)
    pp_name = task.name.lower().replace(" ", "_")
    with open(f"{pp_name}_gp_post_pred.pkl", "wb") as of:
        pickle.dump(pp, of)
    return pp


def plot_posterior_predictive_samples(
    name,
    s_ctx,
    f_ctx,
    valid_len,
    s,
    f,
    f_noisy,
    pp_samples,
    hdi_prob=0.9,
):
    idx = s_ctx.argsort()
    f_hat = np.array(pp_samples)
    f_hat_mu = f_hat.mean(axis=0)
    f_hat_hdi = az.hdi(f_hat)
    plt.plot(s, f, color="black")
    plt.plot(s, f_hat_mu[idx], color="steelblue")
    plt.scatter(s_ctx[:valid_len], f_ctx[:valid_len], color="black")
    plt.fill_between(
        s,
        f_hat_hdi[idx, 0],
        f_hat_hdi[idx, 1],
        alpha=0.4,
        color="steelblue",
        interpolate=True,
    )
    ax = plt.gca()
    ax.set_xlabel("s")
    ax.set_ylabel("f")
    plt.title(f"GP: {name} Posterior Predictive")
    plt.savefig(f"GP: {name} Posterior Predictive.pdf", dpi=600)
    plt.clf()


if __name__ == "__main__":
    main()
