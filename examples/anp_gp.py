#!/usr/bin/env python3
import pickle
from dataclasses import dataclass

import arviz as az
import hydra
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
from jax.tree_util import Partial
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from omegaconf import DictConfig, OmegaConf
from scipy.stats import norm
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


@dataclass
class Task:
    name: str
    kernel: Kernel
    var: Prior
    ls: Prior


@hydra.main("configs", "anp_gp", None)
def main(cfg: DictConfig):
    key = random.key(42)
    OmegaConf.register_new_resolver("eval", eval)
    s = build_grid(cfg.data.grid)
    periodic_0_1 = Partial(periodic, period=cfg.data.period)
    var, ls = Prior("fixed", {"value": 1.0}), Prior("fixed", {"value": 0.2})
    periodic_task = Task(name="Periodic", kernel=periodic_0_1, var=var, ls=ls)
    matern_3_2_task = Task(name="Matern 3-2", kernel=matern_3_2, var=var, ls=ls)
    # for task in [periodic_task, matern_3_2_task]:
    for task in [matern_3_2_task]:
        rng_gp, rng_idx, rng_hmc, rng_tr, rng_pr, rng_eps, key = random.split(key, 7)
        gp = GP(task.kernel, task.var, task.ls)
        _, _, _, f = gp.simulate(rng_gp, s, 1)
        f_noisy = f + cfg.data.obs_noise * random.normal(rng_eps, f.shape)
        gp_model = build_gp_model(task.kernel)
        idx = random.choice(
            rng_idx,
            jnp.arange(s.size),
            shape=(cfg.data.num_test,),
            replace=False,
        )
        # pp = hmc(task, gp_model, rng_hmc, s, f, f_noisy, idx, cfg.infer)
        # plot_posterior_predictive_samples(task.name, s, f, f_noisy, idx, pp["obs"])
        state = train(cfg, gp, s, rng_tr)
        plot_posterior_predictive(task.name, s, f, f_noisy, idx, state, rng_pr)


def build_gp_model(kernel):
    def m(s, f, idx):
        variance = numpyro.sample("variance", dist.HalfNormal())
        lengthscale = numpyro.sample("lengthscale", dist.HalfNormal())
        # jitter added on diagonal for cholesky decomposition stability
        K = kernel(s, s, variance, lengthscale) + 1e-6 * jnp.eye(len(s))
        f_mu = numpyro.sample("f_mu", dist.MultivariateNormal(0, K))
        f_sigma = numpyro.sample("f_sigma", dist.HalfNormal(0.1))
        obs, f_mu = (f, f_mu) if f is None else (f[idx], f_mu[idx])
        numpyro.sample("obs", dist.Normal(f_mu, f_sigma), obs=obs)

    return m


def hmc(task, model, rng, s, f, f_noisy, idx, infer_cfg: DictConfig):
    rng_mcmc, rng_pp = random.split(rng)
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    mcmc = MCMC(nuts, **infer_cfg)
    mcmc.run(rng_mcmc, s, f, idx)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    pp = Predictive(model, samples)(rng_pp, s, None, idx)
    pp.update({"s": s, "f": f, "f_noisy": f_noisy, "idx": idx})
    pp_name = task.name.lower().replace(" ", "_")
    with open(f"{pp_name}_gp_post_pred.pkl", "wb") as of:
        pickle.dump(pp, of)
    return pp


def plot_posterior_predictive_samples(
    model_name,
    s,
    f,
    f_noisy,
    idx,
    pp_samples,
    hdi_prob=0.9,
):
    f_hat = np.array(pp_samples)
    f_hat_mu = f_hat.mean(axis=0)
    f_hat_hdi = az.hdi(f_hat)
    plt.plot(s, f, color="black")
    plt.plot(s, f_hat_mu, color="steelblue")
    plt.scatter(s[idx], f_noisy[idx], color="black")
    plt.fill_between(
        s,
        f_hat_hdi[:, 0],
        f_hat_hdi[:, 1],
        alpha=0.4,
        color="steelblue",
        interpolate=True,
    )
    ax = plt.gca()
    ax.set_xlabel("s")
    ax.set_ylabel("f")
    plt.title(f"GP: {model_name} Posterior Predictive")
    plt.savefig(f"GP: {model_name} Posterior Predictive.pdf", dpi=600)
    plt.clf()


def train(cfg: DictConfig, gp: GP, s: Array, rng: Array):
    rng_model, rng_init, rng_sample, rng_loader, rng_train = random.split(rng, 5)
    model = instantiate(OmegaConf.to_container(cfg.model, resolve=True), rng_model)
    loader = dataloader(rng_loader, gp, s, **cfg.data.loader)
    (s_ctx_init, f_ctx_init, valid_lens), (s_test_init, _) = next(loader)
    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(
            rng_init,
            rng_sample,
            s_ctx_init,
            f_ctx_init,
            s_test_init,
        )["params"],
        tx=optax.adam(1e-3),
        metrics=Metrics.empty(),
    )
    metrics = {"train_loss": []}
    print(model)
    with tqdm(range(1, cfg.train.num_batches + 1), unit="batch") as pbar:
        rng_dropout, rng_sample, rng_train = random.split(rng_train, 3)
        for i in pbar:
            batch = next(loader)
            state = train_step(rng_dropout, rng_sample, state, batch)
            if i % 10 == 0:
                state = compute_metrics(rng_sample, state, batch)
                for metric, value in state.metrics.compute().items():
                    metrics[f"train_{metric}"].append(value)
                state = state.replace(metrics=state.metrics.empty())
                pbar.set_postfix(loss=f"{metrics['train_loss'][-1]:.3f}")
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


# TODO(danj): sample and add correct valid lengths
def dataloader(key, gp, s, batch_size=64, approx=False):
    _s = jnp.repeat(s[None, ...], batch_size, axis=0)
    while True:
        rng, key = random.split(key)
        _var, _ls, _z, f = gp.simulate(rng, s, batch_size, approx)
        valid_lens = jnp.repeat(s.shape[0], batch_size)
        yield (_s, f, valid_lens), (_s, f)


# TODO(danj): implement
def random_subset(rng, s, f):
    pass


@jit
def train_step(rng_dropout, rng_sample, state, batch):
    def loss_fn(params):
        (s_ctx, f_ctx, valid_lens), (s_test, f_test) = batch
        f_mu, f_log_var, _z_mu, _z_log_var = state.apply_fn(
            {"params": params},
            rng_sample,
            s_ctx,
            f_ctx,
            s_test,
            valid_lens,
            training=True,
            rngs={"dropout": rng_dropout},
        )
        return neural_process_maximum_likelihood_loss(f_test, f_mu, f_log_var)

    return state.apply_gradients(grads=grad(loss_fn)(state.params))


@jit
def compute_metrics(rng, state, batch):
    (s_ctx, f_ctx, valid_lens), (s_test, f_test) = batch
    f_mu, f_log_var, _z_mu, _z_log_var = state.apply_fn(
        {"params": state.params}, rng, s_ctx, f_ctx, s_test, valid_lens
    )
    loss = neural_process_maximum_likelihood_loss(f_test, f_mu, f_log_var)
    metric_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metric_updates)
    return state.replace(metrics=metrics)


def plot_posterior_predictive(
    model_name,
    s,
    f,
    f_noisy,
    idx,
    state,
    rng,
    hdi_prob=0.9,
):
    s_test = s[None, ...]
    num_s, num_ctx = s_test.shape[1], len(idx)
    s_ctx, f_ctx = jnp.zeros(s_test.shape), jnp.zeros(s_test.shape)
    s_ctx = s_ctx.at[:, :num_ctx, :].set(s_test[:, idx, :])
    f_ctx = f_ctx.at[:, :num_ctx, :].set(f_noisy[:, idx, :])
    valid_lens = jnp.array([num_ctx])
    f_mu, f_log_var, _z_mu, _z_log_var = state.apply_fn(
        {"params": state.params}, rng, s_ctx, f_ctx, s_test, valid_lens
    )
    # TODO(danj): sample from pp instead??
    f_mu = f_mu.squeeze().mean(axis=0)
    f_std = jnp.exp(f_log_var.squeeze().mean(axis=0) / 2)
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    f_lower, f_upper = f_mu - z_score * f_std, f_mu + z_score * f_std
    s = s.squeeze()
    plt.plot(s, f.squeeze(), color="black")
    plt.plot(s, f_mu, color="steelblue")
    plt.scatter(
        s_ctx[:, :num_ctx, :].squeeze(),
        f_ctx[:, :num_ctx, :].squeeze(),
        color="black",
    )
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
    plt.title(f"ANP: {model_name} Posterior Predictive")
    plt.savefig(f"ANP: {model_name} Posterior Predictive.pdf", dpi=600)
    plt.clf()


if __name__ == "__main__":
    main()
