#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from jax.scipy.stats import norm
from pygifsicle import optimize
from sps.gp import GP
from sps.kernels import rbf
from sps.priors import Prior
from tqdm import tqdm

import dsp.meta_regression.train_utils as tu

# TODO(danj): Plot showing that adding a point to context is Bayesian Updating:
# 1. Create a sample and calculate posterior preditive.
# 2. Add one point from linear interpolation and generate new posterior predictive.
# 3. Update the plot, keeping dimensions fixed.


def main(args):
    rng = random.key(args.seed)
    rng_gp, rng_extra, rng = random.split(rng, 3)
    s_min, s_max, num_ctx, num_test, ls = -2, 2, 10, 128, 0.2
    s_ctx = jnp.linspace(s_min + 0.1, s_max - 0.1, num_ctx)[:, None]  # [L_ctx, 1]
    s_test = jnp.linspace(s_min, s_max, num_test)[:, None]  # [L_test, 1]
    gp = GP(rbf, ls=Prior("fixed", {"value": ls}))
    f, *_ = gp.simulate(rng_gp, jnp.vstack([s_ctx, s_test]), batch_size=1)
    f_ctx, f_test = f[0, :num_ctx, :], f[0, num_ctx:, :]
    state, _ = tu.load_ckpt(Path(args.ckpt_path))
    f_mu, f_std, *_ = state.apply_fn(
        {"params": state.params, **state.kwargs},
        s_ctx[None, ...],  # add batch dim
        f_ctx[None, ...],
        s_test[None, ...],
        rngs={"extra": rng_extra},
    )
    save_samples_gif(
        rng,
        state,
        s_ctx,
        f_ctx,
        s_test,
        f_test,
        f_mu,
        f_std,
        f"samples_{args.seed}.gif",
    )


def save_samples_gif(
    rng: jax.Array,
    state: tu.TrainState,
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    s_test: jax.Array,
    f_test: jax.Array,
    f_mu: jax.Array,
    f_std: jax.Array,
    path: str = "samples_hdi.gif",
    num_samples: int = 512,
    hdi_prob: float = 0.95,
):
    paths = []
    s, f = tu.sample(rng, state, s_ctx, f_ctx, s_test, num_samples)
    s, f = s.squeeze(), f.squeeze()
    s_ctx, f_ctx = s_ctx.squeeze(), f_ctx.squeeze()
    s_test, f_test = s_test.squeeze(), f_test.squeeze()
    f_mu, f_std = f_mu.squeeze(), f_std.squeeze()
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    for i in tqdm(range(num_samples + 1), desc="Samples"):
        f_mu_samples, f_std_samples = f[:i].mean(axis=0), f[:i].std(axis=0)
        fig = tu.plot_posterior_predictive(
            s_ctx, f_ctx, s_test, f_test, f_mu, f_std, hdi_prob
        )
        ax = fig.axes[0]
        ax.fill_between(
            s_test,
            f_mu_samples - z_score * f_std_samples,
            f_mu_samples + z_score * f_std_samples,
            alpha=0.4,
            color="darkorange",
            interpolate=True,
        )
        ax.set_ylim(-3.0, 3.0)
        if i < num_samples:
            ax.plot(s[i], f[i], color="darkorange")
        else:
            ax.plot(s_test, f_mu_samples, color="darkorange")
        fig.suptitle("Samples")
        paths += [f"/tmp/sample_{i}.png"]
        save(fig, paths[-1])
    frames = jnp.stack([iio.imread(p) for p in paths], axis=0)
    iio.imwrite(path, frames)
    optimize(path)


def save(fig, name: str):
    fig.savefig(name, dpi=150)
    plt.clf()
    plt.close(fig)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "ckpt_path",
        help="Path to a 1D GP RBF model checkpoint.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=7,
        type=int,
        help="Root seed for all random operations.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args)
