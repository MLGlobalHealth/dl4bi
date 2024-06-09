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
from tqdm import tqdm

from dsp import MLP, Phi, PiVAE


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    loss_1: metrics.Average.from_output("loss_1")
    loss_2: metrics.Average.from_output("loss_2")
    reg_loss: metrics.Average.from_output("reg_loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


def main(kernel: str, num_batches: int):
    rbf_dim, hidden_dim, beta_dim = 32, 128, 128
    z_dim, loc_dims = 32, (32, 1)
    key = random.key(42)
    rng_data, rng_init, rng_z, rng_train, rng_sample = random.split(key, 5)
    loader = dataloader(rng_data, GP(kernel), loc_dims)
    s, f = next(loader)
    phi = Phi([rbf_dim, hidden_dim, beta_dim])
    encoder = MLP([hidden_dim, z_dim])
    decoder = MLP([hidden_dim, beta_dim])
    model = PiVAE(phi, encoder, decoder, z_dim)
    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(rng_init, rng_z, s, f)["params"],
        tx=optax.adam(1e-3),
        metrics=Metrics.empty(),
    )
    metrics = {"loss": [], "loss_1": [], "loss_2": [], "reg_loss": []}
    with tqdm(range(1, num_batches + 1), unit="batch") as pbar:
        for i in pbar:
            batch = next(loader)
            rng_step, rng_train = random.split(rng_train)
            state = train_step(rng_step, state, batch)
            if i % 100 == 0:
                state = compute_metrics(rng_step, state, batch)
                for metric, value in state.metrics.compute().items():
                    metrics[metric].append(value)
                state = state.replace(metrics=state.metrics.empty())
                pbar.set_postfix(**{k: f"{v[-1]:.3f}" for k, v in metrics.items()})
    s, f = next(loader)
    f_hat_beta, _f_hat_beta_hat, _mu, _log_var = state.apply_fn(
        {"params": state.params}, rng_sample, s, f
    )
    s_5 = s[:5].squeeze().T
    plt.title("f vs f_hat samples")
    plt.plot(s_5, f[:5].squeeze().T, color="black")
    plt.plot(s_5, f_hat_beta[:5].squeeze().T, color="red")
    plt.savefig("pi_vae_f_vs_f_hat.png")


def dataloader(key, gp, loc_dims, batch_size=1024, approx=True):
    """This returns the same batch forever. See `PiVAE` documentation."""
    rng_loc, rng_gp = random.split(key)
    s = random.uniform(rng_loc, (batch_size, *loc_dims)).sort(axis=1)
    f = []
    for i in range(batch_size):
        rng_gp_i, rng_gp = random.split(rng_gp)
        _, _, _, _f = gp.simulate(rng_gp_i, s[i], 1, approx)
        f += [_f.squeeze()]
    f = jnp.array(f)
    while True:
        yield s, f


@jax.jit
def train_step(rng, state, batch):
    def loss_fn(params):
        s, f = batch
        f_hat_beta, f_hat_beta_hat, mu, log_var = state.apply_fn(
            {"params": params}, rng, s, f
        )
        loss_1 = optax.squared_error(f_hat_beta, f).mean()
        loss_2 = optax.squared_error(f_hat_beta_hat, f).mean()
        reg_loss = kl_divergence(mu, log_var)
        return loss_1 + loss_2 + reg_loss

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
    s, f = batch
    f_hat_beta, f_hat_beta_hat, mu, log_var = state.apply_fn(
        {"params": state.params}, rng, s, f
    )
    loss_1 = optax.squared_error(f_hat_beta, f).mean()
    loss_2 = optax.squared_error(f_hat_beta_hat, f).mean()
    reg_loss = kl_divergence(mu, log_var)
    loss = loss_1 + loss_2 + reg_loss
    metric_updates = state.metrics.single_from_model_output(
        loss=loss, loss_1=loss_1, loss_2=loss_2, reg_loss=reg_loss
    )
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
