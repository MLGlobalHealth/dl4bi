import argparse
import sys

import optax
from flax.training.train_state import TrainState
from jax import random
from sps.gp import GP
from sps.utils import build_grid

from dge import DeepChol


def main(kernel: str):
    locations = build_grid([{"start": 0, "stop": 1, "num": 32}])
    key = random.key(42)
    rng_data, rng_init = random.split(key, 2)
    loader = dataloader(rng_data, GP(), locations)
    var, ls, z, f = next(loader)
    state = TrainState.create(
        apply_fn=DeepChol.apply,
        params=DeepChol.init(rng_init, z, var, ls)["params"],
        tx=optax.adam(1e-3),
    )


def dataloader(key, gp, locations, batch_size=1024, approx=True):
    while True:
        rng, key = random.split(key)
        yield gp.simulate(rng, locations, batch_size, approx)


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        var, ls, z, f = batch
        f_hat = state.apply_fn({"params": params}, z, var, ls)
        loss = nn.mse


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-k", "--kernel", default="rbf")
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args.kernel)
