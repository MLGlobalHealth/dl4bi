from functools import partial
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import permutations
import tqdm
from jax import jit, random

Apply = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]
]


def diagonal_sample(
    rng: jax.Array,
    apply: Apply,
    s_ctx: jax.Array,  # [L_ctx, D]
    f_ctx: jax.Array,  # [L_ctx, 1]
    s_test: jax.Array,  # [L_test, D]
    B: int,  # how many paths to sample
):
    L_test, _ = s_test.shape

    # precompute the randomness
    normals = random.normal(rng, (B, L_test, 1))
    log_densities = jnp.sum(jax.scipy.stats.norm.logpdf(normals), axis=1)

    # expand the dimensions to B
    s_ctx = jnp.repeat(s_ctx[None], B, axis=0)
    f_ctx = jnp.repeat(f_ctx[None], B, axis=0)
    s_test = jnp.repeat(s_test[None], B, axis=0)

    f_mu, f_std = apply(s_ctx, f_ctx, s_test)
    f_sampled = normals * f_std + f_mu

    # Jacobian transform eps -> f_sampled
    log_densities -= jnp.log(f_std).sum(axis=1)

    return f_sampled, log_densities.squeeze(-1)


@partial(jit, static_argnames=["apply"])
def autoregressive_sample(
    rng: jax.Array,
    apply: Apply,
    s_ctx: jax.Array,  # [B, L_ctx, D_s]
    f_ctx: jax.Array,  # [B, L_ctx, D_f]
    s_test: jax.Array,  # [B, L_test, D_s]
):
    """
    Implementation of the autoregressive sampling generative model, batched.
    """
    B, L_ctx, _ = s_ctx.shape
    _, L_test, _ = s_test.shape
    _, _, D_f = f_ctx.shape

    s = jnp.concat([s_ctx, s_test], axis=1)
    f = jnp.pad(f_ctx, ((0, 0), (0, L_test), (0, 0)))

    # Note that the independent random normals can be pre-sampled.
    # It doesn't matter whether sampling is done here, or in the for loop,
    # as in each iteration the N(0, 1) sampling is independent
    normals = random.normal(rng, (B, L_test, D_f))
    log_densities = jnp.sum(jax.scipy.stats.norm.logpdf(normals), axis=1)

    def loop(i: int, carry: tuple[jax.Array, jax.Array]):
        f, log_densities = carry
        s_test_i = s_test[:, i][:, None]  # [B, 1, D_s]
        eps = normals[:, i]  # [B, D_f]
        valid_lens_ctx = jnp.repeat(L_ctx + i, B)  # [B]

        f_mu_i, f_std_i = apply(s, f, s_test_i, valid_lens_ctx)
        f_mu_i, f_std_i = f_mu_i.squeeze(1), f_std_i.squeeze(1)  # [B, D_f]
        f_sampled = eps * f_std_i + f_mu_i

        return (
            f.at[:, L_ctx + i].set(f_sampled),
            # Jacobian transform eps -> f_sampled
            # q(f) = q(eps) * |df/deps|^-1 = q(eps) / f_std, hence in log space subtract log(f_std)
            log_densities - jnp.log(f_std_i),
        )

    f, log_densities = jax.lax.fori_loop(
        0,
        L_test,
        loop,
        (f, log_densities),
    )

    return f[:, L_ctx:], log_densities


@partial(jit, static_argnames=["apply"])
def autoregressive_logpdf(
    apply: Apply,
    s_ctx: jax.Array,  # [B, L_ctx, D_s]
    f_ctx: jax.Array,  # [B, L_ctx, D_f]
    s_test: jax.Array,  # [B, L_test, D_s]
    f_test: jax.Array,  # [B, L_test, D_f]
):
    """
    Computes the log-likelihood induced by the autoregressive model, batched.

    Note that it assumes the model where samples are taken in the order given by `s_test, f_test`.
    """
    B, L_test, _ = s_test.shape

    s = jnp.concat([s_ctx, s_test], axis=1)
    f = jnp.concat([f_ctx, f_test], axis=1)

    def fun(valid_lens_ctx, s_test_i, f_test_i):
        f_mu_i, f_std_i = apply(s, f, s_test_i, valid_lens_ctx)
        f_mu_i, f_std_i = f_mu_i.squeeze(1), f_std_i.squeeze(1)  # [B, D_f]
        return jax.scipy.stats.norm.logpdf(f_test_i, f_mu_i, f_std_i)

    log_densities = jax.vmap(fun, in_axes=(1, 1, 1), out_axes=1)(
        jnp.arange(L_test)[None].repeat(B),  # [B, L_test]
        s_test,  # [B, L_test, D_s]
        f_test,  # [B, L_test, D_f]
    )  # -> [B, L_test, D_f]

    return log_densities.sum(axis=1)


def autoregressive_sample_multiple_paths(
    rng: jax.Array,
    apply: Apply,
    s_ctx: jax.Array,  # [L_ctx, D_s]
    f_ctx: jax.Array,  # [L_ctx, D_f]
    s_test: jax.Array,  # [L_test, D_s]
    batch_size: int,
    num_paths: int,
    strategy: Literal["preserve", "ltr", "random", "furthest", "closest"],
):
    """
    Autoregressively sample `num_paths` from the model using the specified strategy,
    and return the paths and associated log-densities.

    `num_paths` will be rounded up to a multiple of `batch_size`.

    NOTE: If `strategy` is "random", the densities will be inaccurate,
    as they don't take into account marginalizing over the permutations.
    """
    num_iters = (num_paths - 1) // batch_size + 1  # ceil division
    all_paths, all_log_densities = [], []

    if strategy != "random":
        match strategy:
            case "preserve":
                idx = idx_inv = ...
            case "closest":
                idx = permutations.closest_first(s_ctx, s_test)
                idx_inv = permutations.invert_permutation(idx)
            case "furthest":
                idx = permutations.furthest_first(s_ctx, s_test)
                idx_inv = permutations.invert_permutation(idx)
            case "ltr":
                idx = permutations.ltr(s_test)
                idx_inv = permutations.invert_permutation(idx)

        s_test = s_test[idx]
        s_test = jnp.repeat(s_test[None], batch_size, axis=0)
        f_ctx = jnp.repeat(f_ctx[None], batch_size, axis=0)
        s_ctx = jnp.repeat(s_ctx[None], batch_size, axis=0)

        for i in tqdm.trange(num_iters, desc=f"Strategy {strategy}"):
            rng, rng_i = random.split(rng)
            paths, log_densities = autoregressive_sample(
                rng_i,
                apply,
                s_ctx,
                f_ctx,
                s_test,
            )
            assert paths.shape == (batch_size, s_test.shape[1], 1)
            all_paths.append(paths)
            all_log_densities.append(log_densities)

        all_paths = jnp.concat(all_paths, axis=0)
        all_log_densities = jnp.concat(all_log_densities, axis=0)

        return all_paths[:, idx_inv], all_log_densities

    else:
        # random strategy needs special handling as we permute randomly each path
        L_test, D_s = s_test.shape
        _, D_f = f_ctx.shape

        # expand the dimensions to batch_size
        s_ctx = jnp.repeat(s_ctx[None], batch_size, axis=0)
        f_ctx = jnp.repeat(f_ctx[None], batch_size, axis=0)
        s_test = jnp.repeat(s_test[None], batch_size, axis=0)

        for i in tqdm.trange(num_iters, desc="Strategy random"):
            rng, rng_perm = jax.random.split(rng)

            # Locations for each path are permuted independently.
            idx = permutations.random_permutations(rng_perm, L_test, batch_size)
            idx_inv = jax.vmap(permutations.invert_permutation)(idx)

            # idx needs to match dimension of array in take_along_axis to permute the data correctly
            idx = jnp.repeat(idx[..., None], D_s, axis=-1)
            idx_inv = jnp.repeat(idx_inv[..., None], D_f, axis=-1)
            assert idx.shape == s_test.shape

            paths, log_densities = autoregressive_sample(
                rng,
                apply,
                s_ctx,
                f_ctx,
                jnp.take_along_axis(s_test, idx, axis=1),
            )

            assert idx_inv.shape == paths.shape
            all_paths.append(jnp.take_along_axis(paths, idx_inv, axis=1))
            all_log_densities.append(log_densities)

        all_paths = jnp.concat(all_paths, axis=0)
        all_log_densities = jnp.concat(all_log_densities, axis=0)

        return all_paths, all_log_densities
