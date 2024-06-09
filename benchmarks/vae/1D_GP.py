#!/usr/bin/env python3
import argparse
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from clu import metrics
from flax import struct
from flax.training import train_state
from jax import random
from jax.scipy.stats.multivariate_normal import logpdf as mvn_logp
from sps.gp import GP
from sps.utils import build_grid
from tqdm import tqdm

from dsp.core import MLP
from dsp.vae import PriorCVAE


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


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
    plt.savefig("f_vs_f_hat.pdf")


def dataloader(key, gp, locations, batch_size=1024, approx=True):
    while True:
        rng, key = random.split(key)
        yield gp.simulate(rng, locations, batch_size, approx)


@jax.jit
def train_step(rng, state, batch):
    def loss_fn(params):
        var, ls, _, f = batch
        f_hat, mu, log_var = state.apply_fn({"params": params}, rng, var, ls, f)
        return neg_elbo(f, f_hat, mu, log_var)

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads)


def neg_elbo(f, f_hat, mu, log_var):
    logp_recon = gaussian_logp(f, f_hat)
    kl_div = kl_divergence(mu, log_var)
    return kl_div - logp_recon


def gaussian_logp(y, y_hat):
    y = y.reshape(y.shape[0], -1)
    y_hat = y_hat.reshape(y_hat.shape[0], -1)
    mu = jnp.zeros(y.shape[1])
    cov = jnp.eye(y.shape[1])
    return mvn_logp(y - y_hat, mu, cov).mean()


def kl_divergence(mu, log_var):
    return (0.5 * (jnp.exp(log_var) + jnp.square(mu) - 1 - log_var)).mean()


@jax.jit
def compute_metrics(rng, state, batch):
    var, ls, _, f = batch
    f_hat, mu, log_var = state.apply_fn({"params": state.params}, rng, var, ls, f)
    loss = neg_elbo(f, f_hat, mu, log_var)
    metric_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metric_updates)
    return state.replace(metrics=metrics)


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
