#!/usr/bin/env python3
import pickle
from collections.abc import Iterable
from dataclasses import dataclass

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
from clu import metrics
from flax import struct
from flax.training import train_state
from jax import Array, grad, jit, random
from jax.scipy.stats import norm
from jax.tree_util import Partial
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from omegaconf import DictConfig, OmegaConf
from sps.gp import GP
from sps.kernels import Kernel, matern_3_2, periodic
from sps.priors import Prior
from sps.utils import build_grid
from tqdm import tqdm

from dge import *


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics
    projections: flax.core.FrozenDict


@dataclass
class Task:
    name: str
    kernel: Kernel
    var: Prior
    ls: Prior


@hydra.main("configs", "hftx_gp", None)
def main(cfg: DictConfig):
    key = random.key(42)
    OmegaConf.register_new_resolver("eval", eval)
    s = build_grid(cfg.data.grid)
    periodic_0_1 = Partial(periodic, period=cfg.data.period)
    var, ls = Prior("fixed", {"value": 1.0}), Prior("fixed", {"value": 0.2})
    periodic_task = Task(name="Periodic", kernel=periodic_0_1, var=var, ls=ls)
    matern_3_2_task = Task(name="Matern 3-2", kernel=matern_3_2, var=var, ls=ls)
    for task in [matern_3_2_task, periodic_task]:
        print(task.name)
        rng_loader, rng_hmc, rng_tr, key = random.split(key, 4)
        gp = GP(task.kernel, task.var, task.ls)
        loader = dataloader(rng_loader, gp, s, **cfg.data.loader)
        (s_ctx, f_ctx, valid_lens), (s_test, f_test, f_noisy) = next(loader)
        state = train(cfg, loader, rng_tr)
        valid_lens = valid_lens.at[0].set(cfg.data.num_test)
        f_mu, f_log_var = state.apply_fn(
            {"params": state.params, "projections": state.projections},
            # {"params": state.params},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens,
        )
        f_mu, f_log_var = f_mu[0].squeeze(), f_log_var[0].squeeze()
        s_ctx, f_ctx = s_ctx[0].squeeze(), f_ctx[0].squeeze()
        s_test, f_test = s_test[0].squeeze(), f_test[0].squeeze()
        f_noisy, valid_len = f_noisy[0].squeeze(), valid_lens[0]
        plot_posterior_predictive_params(
            task.name, s_ctx, f_ctx, valid_len, s_test, f_test, f_noisy, f_mu, f_log_var
        )
        # gp_model = build_gp_model(task.kernel)
        # pp = hmc(task, gp_model, rng_hmc, s_ctx, f_ctx, valid_len, cfg.infer)
        # plot_posterior_predictive_samples(
        #     task.name, s_ctx, f_ctx, valid_len, s, f, f_noisy, pp["obs"]
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
    while True:
        rng_gp, rng_noise, rng_perm, rng_valid, key = random.split(key, 5)
        _var, _ls, _z, f = gp.simulate(rng_gp, s, batch_size, approx)
        f_noisy = f + obs_noise * random.normal(rng_noise, f.shape)
        # permuting same size matrices with the same rng permutes both equivalently
        s_perm = random.permutation(rng_perm, _s, axis=1, independent=True)
        f_noisy_perm = random.permutation(rng_perm, f_noisy, axis=1, independent=True)
        valid_lens = random.randint(rng_valid, (batch_size,), min_obs, max_obs)
        yield (s_perm, f_noisy_perm, valid_lens), (_s, f, f_noisy)


def train(cfg: DictConfig, loader: Iterable, rng: Array):
    rng_model, rng_init, rng_train = random.split(rng, 3)
    model = instantiate(OmegaConf.to_container(cfg.model, resolve=True), rng_model)
    (s_ctx, f_ctx, valid_lens), (s_test, _, _) = next(loader)
    variables = model.init(rng_init, s_ctx, f_ctx, s_test, valid_lens)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        projections=variables["projections"],
        tx=optax.adam(1e-3),
        metrics=Metrics.empty(),
    )
    metrics = {"train_loss": []}
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"{model}\n\nParam count: {param_count}")
    with tqdm(range(1, cfg.train.num_batches + 1), unit="batch") as pbar:
        rng_dropout, rng_train = random.split(rng_train, 2)
        for i in pbar:
            batch = next(loader)
            state = train_step(rng_dropout, state, batch)
            if i % 10 == 0:
                state = compute_metrics(state, batch)
                for metric, value in state.metrics.compute().items():
                    metrics[f"train_{metric}"].append(value)
                state = state.replace(metrics=state.metrics.empty())
                pbar.set_postfix(nll=f"{metrics['train_loss'][-1]:.3f}")
    return state


def instantiate(d: dict, rng: Array):
    for k in d:
        if isinstance(d[k], dict):
            d[k] = instantiate(d[k], rng)
    if "cls" in d:
        if d["cls"] == "GaussianFourierEmbedding":
            embed_dim = d["kwargs"]["embed_dim"]
            input_dim = d["kwargs"]["input_dim"]
            var = d["kwargs"].get("var", 10.0)
            B = random.normal(rng, (embed_dim, input_dim))
            return GaussianFourierEmbedding(B, var)
        else:
            cls, kwargs = d["cls"], d.get("kwargs", {})
            return globals()[cls](**kwargs)
    elif "func" in d:
        return eval(d["func"])
    return d


@jit
def train_step(rng_dropout, state, batch):
    def loss_fn(params):
        (s_ctx, f_ctx, valid_lens), (s_test, f_test, f_test_noisy) = batch
        f_mu, f_log_var = state.apply_fn(
            {"params": params, "projections": state.projections},
            # {"params": params},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens,
            training=True,
            rngs={"dropout": rng_dropout},
        )
        return -norm.logpdf(f_test_noisy, f_mu, jnp.exp(f_log_var / 2)).mean()

    return state.apply_gradients(grads=grad(loss_fn)(state.params))


@jit
def compute_metrics(state, batch):
    (s_ctx, f_ctx, valid_lens), (s_test, f_test, f_test_noisy) = batch
    f_mu, f_log_var = state.apply_fn(
        {"params": state.params, "projections": state.projections},
        # {"params": state.params},
        s_ctx,
        f_ctx,
        s_test,
        valid_lens,
    )
    nll = -norm.logpdf(f_test_noisy, f_mu, jnp.exp(f_log_var / 2)).mean()
    metric_updates = state.metrics.single_from_model_output(loss=nll)
    metrics = state.metrics.merge(metric_updates)
    return state.replace(metrics=metrics)


def plot_posterior_predictive_params(
    name,
    s_ctx,
    f_ctx,
    valid_len,
    s,
    f,
    f_noisy,
    f_mu,
    f_log_var,
    hdi_prob=0.9,
):
    f_std = jnp.exp(f_log_var / 2)
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    f_lower, f_upper = f_mu - z_score * f_std, f_mu + z_score * f_std
    plt.plot(s, f, color="black")
    plt.plot(s, f_mu, color="steelblue")
    plt.scatter(s_ctx[:valid_len], f_ctx[:valid_len], color="black")
    plt.fill_between(
        s,
        f_lower,
        f_upper,
        alpha=0.4,
        color="steelblue",
        interpolate=True,
    )
    ax = plt.gca()
    ax.set_xlabel("s")
    ax.set_ylabel("f")
    plt.title(f"HFTx: {name} Posterior Predictive")
    plt.savefig(f"HFTx: {name} Posterior Predictive.pdf", dpi=600)
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
