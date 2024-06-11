#!/usr/bin/env python3
import argparse
import sys
from collections.abc import Callable
from typing import Iterable

import flax.linen as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import optax
from flax.training.train_state import TrainState
from jax import Array, grad, jit, random
from jax.scipy.stats import norm
from sps.gp import GP
from sps.kernels import rbf
from sps.priors import Prior
from sps.utils import build_grid
from tqdm import tqdm

from dsp.core import MLP
from dsp.vae import DeepChol, PiVAE, PriorCVAE


def dataloader(rng: Array, gp: GP, s: Array, batch_size=1024, approx=True):
    while True:
        rng_batch, rng = random.split(rng)
        yield gp.simulate(rng_batch, s, batch_size, approx)


def train(
    model: nn.Module,
    train_step: Callable,
    num_batches: int = 1000,
    num_s: int = 64,
    seed: int = 0,
):
    key = random.key(seed)
    rng_data, rng_init, rng_z, rng_train = random.split(key, 4)
    var = Prior("fixed", {"value": 1.0})
    ls = Prior("beta", {"a": 3, "b": 7})
    s = build_grid([{"start": 0, "stop": 1, "num": num_s}])
    loader = dataloader(rng_data, GP(rbf, var, ls), s)
    var, ls, _z, f = next(loader)
    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(rng_init, rng_z, var, ls, f)["params"],
        tx=optax.adam(1e-3),
    )
    for i in (pbar := tqdm(range(1, num_batches + 1), unit="batch")):
        batch = next(loader)
        rng_step, rng_train = random.split(rng_train)
        state, loss = train_step(rng_step, state, batch)
        pbar.set_postfix(loss=f"{loss:0.3f}")
    return state


def build_prior_cvae(num_s: int = 64, z_dim: int = 64):
    encoder = MLP([128, z_dim])
    decoder = MLP([128, num_s])
    return PriorCVAE(encoder, decoder, z_dim)


@jit
def prior_cvae_train_step(rng, state, batch):
    def elbo_loss(params):
        var, ls, _z, f = batch
        f_hat, z_mu, z_log_var = state.apply_fn({"params": params}, rng, var, ls, f)
        kl_div = (0.5 * (jnp.exp(z_log_var) + jnp.square(z_mu) - 1 - z_log_var)).mean()
        logp = norm.logpdf(f, f_hat, 1.0).mean()
        return -logp + kl_div

    return state.apply_gradients(grads=grad(elbo_loss)(state.params))


def plot_samples(rng: Array, s: Array, state: TrainState, loader: Iterable):
    var, ls, _, f = next(loader)
    f_hat, _, _ = state.apply_fn({"params": state.params}, rng, var, ls, f)
    _s = s.squeeze()
    plt.title("f vs. f_hat samples")
    plt.plot(_s, f[:5].squeeze().T, color="black")
    plt.plot(_s, f_hat[:5].squeeze().T, color="red")


def simple_model(x1: Array, x2: Array, s: Array, f: Array):
    alpha = numpyro.sample("alpha", dist.Normal(0, 1))
    beta = numpyro.sample("beta", dist.Normal(0, 1))
    var = numpyro.sample("var", dist.HalfNormal(1))
    ls = numpyro.sample("ls", dist.Beta(3, 7))
    f_mu = alpha * x1 + beta * x2
    f_cov = rbf(s, s, var, ls)
    idx = jnp.isfinite(f)  # don't use NaNs in likelihood
    numpyro.sample("f_hat", dist.MultivariateNormal(f_mu, f_cov)[idx], obs=f[idx])


def main(kernel: str, num_batches: int):
    f_dim, z_dim = 32, 32
    locations = build_grid([{"start": 0, "stop": 1, "num": f_dim}])
    key = random.key(42)
    rng_data, rng_init, rng_z, rng_train, rng_sample = random.split(key, 5)
    loader = dataloader(rng_data, GP(kernel), locations)
    var, ls, _, f = next(loader)
    encoder = MLP([128, z_dim])
    decoder = MLP([128, f_dim])
    model = PriorCVAE(encoder, decoder, z_dim)
    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(rng_init, rng_z, var, ls, f)["params"],
        tx=optax.adam(1e-3),
        metrics=Metrics.empty(),
    )
    metrics = {"train_loss": []}
    with tqdm(range(1, num_batches + 1), unit="batch") as pbar:
        for i in pbar:
            batch = next(loader)
            rng_step, rng_train = random.split(rng_train)
            state = train_step(rng_step, state, batch)
            if i % 100 == 0:
                state = compute_metrics(rng_step, state, batch)
                for metric, value in state.metrics.compute().items():
                    metrics[f"train_{metric}"].append(value)
                state = state.replace(metrics=state.metrics.empty())
                pbar.set_postfix(loss=f"{metrics['train_loss'][-1]:.3f}")
    var, ls, _, f = next(loader)
    f_hat, _, _ = state.apply_fn({"params": state.params}, rng_sample, var, ls, f)
    x = jnp.linspace(0, 1, f_dim)
    plt.title("f vs f_hat samples")
    plt.plot(x, f[:5].squeeze().T, color="black")
    plt.plot(x, f_hat[:5].squeeze().T, color="red")
    plt.savefig("prior_cvae_f_vs_f_hat.png")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-k", "--kernel", default="matern_3_2")
    parser.add_argument("-n", "--num_batches", default=10000, type=int)
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args.kernel, args.num_batches)
