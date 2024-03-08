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
from sps.gp import GP
from sps.utils import build_grid
from tqdm import tqdm

from dge import MLP, DeepChol


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


def main(kernel: str, num_batches: int):
    locations = build_grid([{"start": 0, "stop": 1, "num": 32}])
    key = random.key(42)
    rng_data, rng_init = random.split(key, 2)
    loader = dataloader(rng_data, GP(kernel), locations)
    var, ls, z, f = next(loader)
    model = DeepChol(MLP([128, 32]))
    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(rng_init, z, var, ls)["params"],
        tx=optax.adam(1e-3),
        metrics=Metrics.empty(),
    )
    metrics = {"train_loss": []}
    with tqdm(range(1, num_batches + 1), unit="batch") as pbar:
        for i in pbar:
            batch = next(loader)
            state = train_step(state, batch)
            if i % 100 == 0:
                state = compute_metrics(state, batch)
                for metric, value in state.metrics.compute().items():
                    metrics[f"train_{metric}"].append(value)
                state = state.replace(metrics=state.metrics.empty())
                pbar.set_postfix(loss=f"{metrics['train_loss'][-1]:.3f}")
    var, ls, z, f = next(loader)
    f_hat = state.apply_fn({"params": state.params}, z, var, ls)
    x = jnp.linspace(0, 1, 32)
    plt.title("f vs f_hat samples")
    plt.plot(x, f[:5].squeeze().T, color="black")
    plt.plot(x, f_hat[:5].T, color="red")
    plt.savefig("deep_chol_f_vs_f_hat.png")


def dataloader(key, gp, locations, batch_size=1024, approx=True):
    while True:
        rng, key = random.split(key)
        yield gp.simulate(rng, locations, batch_size, approx)


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        var, ls, z, f = batch
        f_hat = state.apply_fn({"params": params}, z, var, ls)
        return optax.squared_error(f_hat, f.squeeze()).mean()

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads)


@jax.jit
def compute_metrics(state, batch):
    var, ls, z, f = batch
    f_hat = state.apply_fn({"params": state.params}, z, var, ls)
    loss = optax.squared_error(f_hat, f.squeeze()).mean()
    metric_updates = state.metrics.single_from_model_output(f_hat=f_hat, f=f, loss=loss)
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
