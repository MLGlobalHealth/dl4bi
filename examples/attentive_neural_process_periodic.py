#!/usr/bin/env python3
import argparse
import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from clu import metrics
from flax import struct
from flax.training import train_state
from jax import random
from jax.scipy.stats import norm
from tqdm import tqdm

from dge import (
    MLP,
    AdditiveScorer,
    AttentiveNeuralProcess,
    DotScorer,
    FixedSinusoidalEmbedding,
    GaussianFourierEmbedding,
    LearnableEmbedding,
    MultiheadAttention,
    NeRFEmbedding,
    TransformerEncoder,
)


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


def main(
    key,
    func,
    pos_embed,
    scorer,
    p_dropout,
    embed_dim,
    num_batches,
    batch_size,
):
    max_x, num_context, num_test, period = 200, 50, 50, 15
    rng_embed, rng_data, rng_init, rng_sample, rng_train = random.split(key, 5)
    s = jnp.linspace(0.0, max_x, num=max_x * 10)[..., None]
    f = func(s / period)
    loader = dataloader(rng_data, s, f, num_context, num_test, batch_size)
    (s_ctx, f_ctx), (s_test, f_test) = next(loader)
    embed_s = LearnableEmbedding(
        get_embedder(pos_embed, rng_embed, embed_dim, 1),
        MLP([embed_dim, embed_dim]),
    )
    embed_s_and_f = LearnableEmbedding(
        get_embedder(pos_embed, rng_embed, embed_dim, 2),
        MLP([embed_dim, embed_dim]),
    )
    enc_ctx_local = TransformerEncoder(scorer.copy())
    enc_ctx_global = TransformerEncoder(scorer.copy())
    # TODO(danj): add post cross-attn linear layer like paper?
    cross_attn = MultiheadAttention(scorer.copy())
    # TODO(danj): original paper has these as the same network
    dec_z_mu = MLP([embed_dim, embed_dim])
    dec_z_log_var = MLP([embed_dim, embed_dim])
    dec_f_mu = MLP([embed_dim * 3, embed_dim * 2, embed_dim, 1], p_dropout=0.0)
    dec_f_log_var = MLP([embed_dim * 3, embed_dim * 2, embed_dim, 1], p_dropout=0.0)
    m = AttentiveNeuralProcess(
        embed_s,
        embed_s_and_f,
        enc_ctx_local,
        enc_ctx_global,
        cross_attn,
        dec_z_mu,
        dec_z_log_var,
        dec_f_mu,
        dec_f_log_var,
    )
    state = TrainState.create(
        apply_fn=m.apply,
        params=m.init(rng_init, rng_sample, s_ctx, f_ctx, s_test)["params"],
        tx=optax.adam(1e-3),
        metrics=Metrics.empty(),
    )
    metrics = {"train_loss": []}
    with tqdm(range(1, num_batches + 1), unit="batch") as pbar:
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
    (s_ctx, f_ctx), _ = next(loader)
    s_ctx, f_ctx = s_ctx[[0], ...], f_ctx[[0], ...]
    s_test = jnp.array([[[732], [828], [987]]])
    f_test = func(s_test / period)
    _, f_ctx_mu, f_ctx_log_var = state.apply_fn(
        {"params": state.params}, rng_sample, s_ctx, f_ctx, s_ctx
    )
    _, f_test_mu, f_test_log_var = state.apply_fn(
        {"params": state.params}, rng_sample, s_ctx, f_ctx, s_test
    )
    s_all = jnp.linspace(0.0, 1000, num=10000)
    f_all = func(s_all / period)
    plt.plot(s_all.squeeze(), f_all.squeeze())
    plt.scatter(s_ctx.squeeze(), f_ctx.squeeze(), color="black", alpha=0.5)
    plt.scatter(s_ctx.squeeze(), f_ctx_mu.squeeze(), color="red", alpha=0.5)
    plt.scatter(s_test.squeeze(), f_test.squeeze(), color="green", alpha=0.5)
    plt.scatter(s_test.squeeze(), f_test_mu.squeeze(), color="red", alpha=0.5)
    plt.title("f_test vs f_test_hat samples")
    plt.savefig(f"{pos_embed}.pdf")


def dataloader(key, s, f, num_context, num_test, batch_size=128, eps=0.1):
    idxs = jnp.arange(len(s))
    while True:
        s_ctx, f_ctx, s_test, f_test = [], [], [], []
        for b in range(batch_size):
            rng, rng_eps, key = random.split(key, 3)
            idx = random.choice(
                rng, idxs, shape=(num_context + num_test,), replace=False
            )
            _f = f + eps * random.normal(rng_eps, f.shape)
            context_idx, test_idx = idx[:num_context], idx[num_context:]
            s_ctx += [s[context_idx]]
            f_ctx += [_f[context_idx]]
            s_test += [s[test_idx]]
            f_test += [_f[test_idx]]
        yield (
            (jnp.stack(s_ctx), jnp.stack(f_ctx)),
            (jnp.stack(s_test), jnp.stack(f_test)),
        )


@jax.jit
def train_step(rng_dropout, rng_sample, state, batch):
    def loss_fn(params):
        (s_ctx, f_ctx), (s_test, f_test) = batch
        zs_global, f_mu, f_log_var = state.apply_fn(
            {"params": params},
            rng_sample,
            s_ctx,
            f_ctx,
            s_test,
            valid_lens=None,
            training=True,
            rngs={"dropout": rng_dropout},
        )
        nll = -norm.logpdf(f_test, f_mu, jnp.exp(f_log_var / 2))
        return nn.relu(nll).sum()  # relu for when variance collapses

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads)


@jax.jit
def compute_metrics(rng_sample, state, batch):
    (s_ctx, f_ctx), (s_test, f_test) = batch
    zs_global, f_mu, f_log_var = state.apply_fn(
        {"params": state.params}, rng_sample, s_ctx, f_ctx, s_test
    )
    nll = -norm.logpdf(f_test, f_mu, jnp.exp(f_log_var / 2))
    loss = nn.relu(nll).sum()
    metric_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metric_updates)
    return state.replace(metrics=metrics)


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


def get_embedder(name: str, key: jax.Array, embed_dim: int = 64, feature_dim: int = 1):
    match name:
        case "identity":
            return jax.jit(lambda s: s)
        case "sinusoidal":
            return FixedSinusoidalEmbedding(embed_dim // feature_dim)
        case "nerf":
            return NeRFEmbedding(embed_dim // feature_dim)
        case "fourier":
            B = random.normal(key, (embed_dim // 2, feature_dim))
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
    parser.add_argument("-e", "--pos_embed", default="sinusoidal")
    parser.add_argument("-d", "--embed_dim", type=int, default=128)
    parser.add_argument("-p", "--p_dropout", type=float, default=0.5)
    parser.add_argument("-n", "--num_batches", type=int, default=500)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    key = random.key(args.key)
    func = get_func(args.func)
    scorer = get_scorer(args.scorer)
    main(
        key,
        func,
        args.pos_embed,
        scorer,
        args.p_dropout,
        args.embed_dim,
        args.num_batches,
        args.batch_size,
    )
