#!/usr/bin/env python3
import argparse
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import jit, random, value_and_grad
from sps import kernels
from sps.gp import GP
from tqdm import tqdm

from dsp.core import MLP, TrainState
from dsp.vae import Phi, PiVAE


def main(kernel_name: str, num_batches: int):
    rbf_dim, hidden_dim, beta_dim = 32, 128, 128
    z_dim, loc_dims = 32, (32, 1)
    key = random.key(42)
    rng_data, rng_params, rng_z, rng_train, rng_sample = random.split(key, 5)
    kernel = getattr(kernels, kernel_name)
    loader = dataloader(rng_data, GP(kernel), loc_dims)
    s, f = next(loader)
    phi = Phi([rbf_dim, hidden_dim, beta_dim])
    encoder = MLP([hidden_dim, z_dim])
    decoder = MLP([hidden_dim, beta_dim])
    model = PiVAE(phi, encoder, decoder, z_dim)
    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init({"params": rng_params, "latent_z": rng_z}, s, f)["params"],
        tx=optax.adam(1e-3),
    )
    with tqdm(range(1, num_batches + 1), unit="batch") as pbar:
        for i in pbar:
            batch = next(loader)
            rng_step, rng_train = random.split(rng_train)
            state, loss = train_step(rng_step, state, batch)
            pbar.set_postfix(loss=f"{loss:0.3f}")
    s, f = next(loader)
    f_hat_beta, _f_hat_beta_hat, _mu, _log_var = state.apply_fn(
        {"params": state.params}, s, f, rngs={"latent_z": rng_z}
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


@jit
def train_step(rng, state, batch):
    def loss_fn(params):
        s, f = batch
        f_hat_beta, f_hat_beta_hat, z_mu, z_std = state.apply_fn(
            {"params": params}, s, f, rngs={"latent_z": rng}
        )
        loss_1 = optax.squared_error(f_hat_beta, f).mean()
        loss_2 = optax.squared_error(f_hat_beta_hat, f).mean()
        kl_div = -jnp.log(z_std) + (z_std**2 + z_mu**2 - 1) / 2
        return loss_1 + loss_2 + kl_div.mean()

    loss, grads = value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


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
