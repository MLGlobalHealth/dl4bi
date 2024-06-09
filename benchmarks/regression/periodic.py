#!/usr/bin/env python3
import argparse
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import jit, random
from tqdm import tqdm

from dsp.core import (
    AdditiveScorer,
    DotScorer,
    FixedSinusoidalEmbedding,
    GaussianFourierEmbedding,
    NeRFEmbedding,
    TrainState,
)
from dsp.regression import KernelRegressor


def main(key, func, embedder, scorer, p_dropout):
    num_batches, batch_size = 100, 64
    rng_data, rng_init, rng_dropout = random.split(key, 3)
    max_x, num_context, num_test = 200, 50, 50
    s = jnp.linspace(0.0, max_x, num=max_x * 10)[..., None]
    period = 15  # lengthen period for graphics
    f = func(s / period)
    loader = dataloader(key, s, f, num_context, num_test, batch_size)
    (s_ctx, f_ctx), (s_test, f_test) = next(loader)
    m = KernelRegressor(embedder, scorer, p_dropout)
    state = TrainState.create(
        apply_fn=m.apply,
        params=m.init(rng_init, s_ctx, f_ctx, s_test)["params"],
        tx=optax.adam(1e-3),
    )
    with tqdm(range(1, num_batches + 1), unit="batch") as pbar:
        _rng_dropout, rng_dropout = random.split(rng_dropout)
        for i in pbar:
            batch = next(loader)
            state = train_step(_rng_dropout, state, batch)
            if i % 10 == 0:
                loss = compute_metrics(state, batch)
                pbar.set_postfix(loss=f"{loss:.3f}")
    (s_ctx, f_ctx), _ = next(loader)
    s_ctx, f_ctx = s_ctx[[0], ...], f_ctx[[0], ...]
    s_test = jnp.array([[[732], [828], [987]]])
    f_test = func(s_test / period)
    f_ctx_hat = state.apply_fn({"params": state.params}, s_ctx, f_ctx, s_ctx)
    f_test_hat = state.apply_fn({"params": state.params}, s_ctx, f_ctx, s_test)
    s_all = jnp.linspace(0.0, 1000, num=10000)
    f_all = func(s_all / period)
    plt.plot(s_all, f_all)
    plt.scatter(s_ctx, f_ctx, color="black", alpha=0.5)
    plt.scatter(s_ctx, f_ctx_hat, color="red", alpha=0.5)
    plt.scatter(s_test, f_test, color="green", alpha=0.5)
    plt.scatter(s_test, f_test_hat, color="red", alpha=0.5)
    plt.title("f_test vs f_test_hat samples")
    plt.savefig(f"{embedder.__class__.__name__}.pdf")


def dataloader(key, s, f, num_context, num_test, batch_size):
    idxs = jnp.arange(len(s))
    while True:
        s_ctx, f_ctx, s_test, f_test = [], [], [], []
        for b in range(batch_size):
            rng, key = random.split(key)
            idx = random.choice(
                rng, idxs, shape=(num_context + num_test,), replace=False
            )
            context_idx, test_idx = idx[:num_context], idx[num_context:]
            s_ctx += [s[context_idx]]
            f_ctx += [f[context_idx]]
            s_test += [s[test_idx]]
            f_test += [f[test_idx]]
        yield (
            (jnp.stack(s_ctx), jnp.stack(f_ctx)),
            (jnp.stack(s_test), jnp.stack(f_test)),
        )


@jit
def train_step(rng_dropout, state, batch):
    def loss_fn(params):
        (s_ctx, f_ctx), (s_test, f_test) = batch
        f_test_hat = state.apply_fn(
            {"params": params},
            s_ctx,
            f_ctx,
            s_test,
            training=True,
            rngs={"dropout": rng_dropout},
        )
        return optax.squared_error(f_test_hat, f_test).mean()

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads)


@jit
def compute_metrics(state, batch):
    (s_ctx, f_ctx), (s_test, f_test) = batch
    f_test_hat = state.apply_fn({"params": state.params}, s_ctx, f_ctx, s_test)
    return optax.squared_error(f_test_hat, f_test).mean()


def get_func(name: str):
    match name:
        case "sine":
            return jnp.sin
        case "cosine":
            return jnp.cos
    raise ValueError(f"Invalid function: {name}")


def get_scorer(name: str):
    match name:
        case "dot":
            return DotScorer()
        case "additive":
            return AdditiveScorer()
    raise ValueError(f"Invalid scorer: {name}")


def get_embedder(name: str, key: jax.Array, embed_dim: int = 64):
    match name:
        case "sinusoidal":
            return FixedSinusoidalEmbedding(embed_dim)
        case "nerf":
            return NeRFEmbedding(embed_dim)
        case "fourier":
            B = random.normal(key, (embed_dim, 1))
            return GaussianFourierEmbedding(B)
    raise ValueError(f"Invalid embedder: {name}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-k", "--key", type=int, default=42)
    parser.add_argument("-f", "--func", default="sine")
    parser.add_argument("-s", "--scorer", default="dot")
    parser.add_argument("-e", "--embedder", default="sinusoidal")
    parser.add_argument("-d", "--embed_dim", type=int, default=64)
    parser.add_argument("-p", "--p_dropout", type=float, default=0.5)
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    key = random.key(args.key)
    func = get_func(args.func)
    embedder = get_embedder(args.embedder, key, args.embed_dim)
    scorer = get_scorer(args.scorer)
    main(key, func, embedder, scorer, args.p_dropout)
