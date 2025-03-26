import os
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import tqdm
from jax import jit, random

from dl4bi.core.utils import concatenate_ctx_and_test
from dl4bi.meta_learning.train_utils import TrainState

from .strategy import Strategy

DEBUG = bool(os.environ.get("DEBUG", False))


def dump_log_densities(log_densities):
    results_dir = os.environ.get("RESULTS_DIR")
    path = Path(results_dir) / f"log_densities_{datetime.now()}.npy"
    jnp.save(path, log_densities)


@dataclass(frozen=True)
class ARSampler:
    model: Callable

    @classmethod
    def from_state(cls, state: TrainState):
        def apply(
            s_ctx: jax.Array,
            f_ctx: jax.Array,
            s_test: jax.Array,
            valid_lens_ctx: jax.Array | None = None,
        ) -> jax.Array:
            return state.apply_fn(
                {"params": state.params, **state.kwargs},
                s_ctx,
                f_ctx,
                s_test,
                valid_lens_ctx,
                training=False,
                # this is not used for TNP-KR, removed so that it is not necessary to pass rng to apply
                # rngs={"extra": rng_extra},
            )

        apply = jit(apply)

        return cls(apply)

    @partial(jit, static_argnums=0)
    def _sample(
        self,
        rng: jax.Array,
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
        log_densities = jax.scipy.stats.norm.logpdf(normals).sum(axis=-1)

        def loop(i: int, carry: tuple[jax.Array, jax.Array]):
            f, log_densities = carry
            s_test_i = s_test[:, i][:, None]  # [B, 1, D_s]
            eps = normals[:, i]  # [B, D_f]
            valid_lens_ctx = jnp.repeat(L_ctx + i, B)  # [B]

            f_mu_i, f_std_i = self.model(s, f, s_test_i, valid_lens_ctx)
            f_mu_i, f_std_i = f_mu_i.squeeze(1), f_std_i.squeeze(1)  # [B, D_f]
            f_sampled = eps * f_std_i + f_mu_i

            return (
                f.at[:, L_ctx + i].set(f_sampled),
                log_densities - jnp.log(f_std_i),  # Jacobian correction
            )

        f, log_densities = jax.lax.fori_loop(
            0,
            L_test,
            loop,
            (f, log_densities),
        )

        return f[:, L_ctx:], log_densities

    @partial(jit, static_argnames=["self", "strategy"])
    def sample(
        self,
        rng: jax.Array,
        s_ctx: jax.Array,  # [B, L_ctx, D_s]
        f_ctx: jax.Array,  # [B, L_ctx, D_f]
        s_test: jax.Array,  # [B, L_test, D_s]
        strategy: Strategy | None = None,
    ):
        """
        Autoregressive sampling with a choice of strategy.
        Returns induced log-densities along with the sample paths.

        NOTE: For the "random" strategy the densities will be incorrect.
        """
        if strategy is None:
            return self._sample(
                rng,
                s_ctx,
                f_ctx,
                s_test,
            )
        else:
            B, L_test, D_s = s_test.shape
            _, _, D_f = f_ctx.shape

            rng, rng_perm = jax.random.split(rng)

            idx, idx_inv = strategy.get_batch_permutation(
                rng, s_ctx, s_test, return_inverse=True
            )
            idx = idx[..., None].repeat(D_s, axis=-1)
            idx_inv = idx_inv[..., None].repeat(D_f, axis=-1)

            assert idx.shape == (B, L_test, D_s)
            assert idx_inv.shape == (B, L_test, D_f)

            s_test = jnp.take_along_axis(s_test, idx, axis=1)
            f_sampled, log_densities = self._sample(
                rng,
                s_ctx,
                f_ctx,
                s_test,
            )
            f_sampled = jnp.take_along_axis(f_sampled, idx_inv, axis=1)
            return f_sampled, log_densities

    def sample_multiple_paths(
        self,
        rng: jax.Array,
        s_ctx: jax.Array,  # [L_ctx, D_s]
        f_ctx: jax.Array,  # [L_ctx, D_f]
        s_test: jax.Array,  # [L_test, D_s]
        batch_size: int,
        num_paths: int,
        strategy: Strategy | None,
    ):
        """
        Autoregressively sample `num_paths` from the model using the specified strategy.

        `num_paths` will be rounded up to a multiple of `batch_size`.

        More efficient than calling `autoregressive_sample` multiple times
        as it permutes the locations only once (except for the random strategy).
        """
        num_iters = (num_paths - 1) // batch_size + 1  # ceil division
        all_paths = []
        all_densities = []

        if strategy != "random":
            if strategy is None:
                idx = idx_inv = ...
            else:
                idx, idx_inv = strategy.get_permutation(
                    rng,  # note rng is not used here since the strategies are deterministic
                    s_ctx,
                    s_test,
                    return_inverse=True,
                )

            s_test = s_test[idx]
            s_test = jnp.repeat(s_test[None], batch_size, axis=0)
            f_ctx = jnp.repeat(f_ctx[None], batch_size, axis=0)
            s_ctx = jnp.repeat(s_ctx[None], batch_size, axis=0)

            for i in tqdm.trange(num_iters, desc=f"Strategy {strategy}"):
                rng, rng_i = random.split(rng)
                paths, log_densities = self._sample(
                    rng_i,
                    s_ctx,
                    f_ctx,
                    s_test,
                )
                assert paths.shape == (batch_size, s_test.shape[1], 1)
                all_paths.append(paths)
                all_densities.append(log_densities)

            all_paths = jnp.concat(all_paths, axis=0)
            all_densities = jnp.concat(all_densities, axis=0)
            return all_paths[:, idx_inv], all_densities

        else:
            # We permute each path independently
            # so this is effectively a call to `sample` with 'random' strategy

            s_ctx = jnp.repeat(s_ctx[None], batch_size, axis=0)
            f_ctx = jnp.repeat(f_ctx[None], batch_size, axis=0)
            s_test = jnp.repeat(s_test[None], batch_size, axis=0)

            for i in tqdm.trange(num_iters, desc="Strategy random"):
                rng, rng_i = random.split(rng)
                paths, log_densities = self.sample(
                    rng_i,
                    s_ctx,
                    f_ctx,
                    s_test,
                    "random",
                )
                all_paths.append(paths)
                all_densities.append(log_densities)

            all_paths = jnp.concat(all_paths, axis=0)
            all_densities = jnp.concatenate(all_densities, axis=0)
            return all_paths, all_densities

    @partial(jit, static_argnums=0)
    def logpdf_diagonal(self, s_ctx, f_ctx, s_test, f_test, valid_lens_ctx):
        mu, std = self.model(s_ctx, f_ctx, s_test, valid_lens_ctx)
        return jax.scipy.stats.norm.logpdf(f_test, mu, std).sum(axis=1)

    @partial(jit, static_argnums=0)
    def logpdf_one_point(
        self,
        s_ctx: jax.Array,  # [B, L_ctx, D_s]
        f_ctx: jax.Array,  # [B, L_ctx, D_f]
        s_test_i: jax.Array,  # [B, D_s]
        f_test_i: jax.Array,  # [B, D_f]
        valid_lens_ctx: jax.Array,  # [B]
    ):
        """
        Log-likelihood of a single test point, batched.
        """
        s_test_i = s_test_i[:, None]  # [B, 1, D_s]
        f_test_i = f_test_i[:, None]  # [B, 1, D_s]
        return self.logpdf_diagonal(s_ctx, f_ctx, s_test_i, f_test_i, valid_lens_ctx)

    @partial(jit, static_argnums=0)
    def _logpdf(
        self,
        s_ctx: jax.Array,  # [B, L_ctx, D_s]
        f_ctx: jax.Array,  # [B, L_ctx, D_f]
        s_test: jax.Array,  # [B, L_test, D_s]
        f_test: jax.Array,  # [B, L_test, D_f]
        valid_lens_ctx: jax.Array,  # [B]
    ) -> jax.Array:
        """
        Computes the log-likelihood induced by the autoregressive model, batched.

        Assumes the model where samples are taken in the order given by `s_test`, `f_test`.
        """
        B, L_test, D_f = f_test.shape

        s = concatenate_ctx_and_test(s_ctx, s_test, valid_lens_ctx)
        f = concatenate_ctx_and_test(f_ctx, f_test, valid_lens_ctx)

        def fun(i, log_densities):
            s_test_i = s_test[:, i]
            f_test_i = f_test[:, i]
            valid_lens_ctx_i = valid_lens_ctx + i
            return log_densities + self.logpdf_one_point(
                s, f, s_test_i, f_test_i, valid_lens_ctx_i
            )

        return jax.lax.fori_loop(0, L_test, fun, jnp.zeros((B, D_f)))

    @partial(jit, static_argnums=0)
    def _logpdf_random(
        self,
        rng: jax.Array,
        s_ctx: jax.Array,  # [B, L_ctx, D_s]
        f_ctx: jax.Array,  # [B, L_ctx, D_f]
        s_test: jax.Array,  # [B, L_test, D_s]
        f_test: jax.Array,  # [B, L_test, D_f]
        valid_lens_ctx: jax.Array,  # [B]
    ):
        _, L_test, _ = s_test.shape

        # TODO @pgrynfelder: do I need to worry about the corellation introduced by permuting the test locations the same way within batch for the NLL estimation?
        idx = random.permutation(rng, L_test)
        s_test = s_test[:, idx]
        f_test = f_test[:, idx]
        return self._logpdf(s_ctx, f_ctx, s_test, f_test, valid_lens_ctx)

    @partial(jit, static_argnames=["self", "M", "Mb"])
    def _logpdf_random_estimate(
        self,
        rng: jax.Array,
        s_ctx: jax.Array,  # [B, L_ctx, D_s]
        f_ctx: jax.Array,  # [B, L_ctx, D_f]
        s_test: jax.Array,  # [B, L_test, D_s]
        f_test: jax.Array,  # [B, L_test, D_f]
        valid_lens_ctx: jax.Array,  # [B]
        M: int,  # number of samples for Monte Carlo estimation
        Mb: int | None,  # batch size for the MC estimate
    ):
        """
        Monte Carlo estimate of the log-likelihood induced by the autoregressive model with random reordering.

        The same permutation is applied to all items in the batch.
        Therefore different paths of interest should be within one batch.
        """

        def fun(rng):
            return self._logpdf_random(
                rng, s_ctx, f_ctx, s_test, f_test, valid_lens_ctx
            )

        log_densities = jax.lax.map(fun, random.split(rng, M), batch_size=Mb)

        # TODO @pgrynfelder: make this nicer
        if DEBUG:
            dump_log_densities(log_densities)

        return jax.nn.logsumexp(log_densities, axis=0) - jnp.log(M)

    def logpdf(
        self,
        rng: jax.Array,  # needed for the random strategy
        s_ctx: jax.Array,  # [B, L_ctx, D_s]
        f_ctx: jax.Array,  # [B, L_ctx, D_f]
        s_test: jax.Array,  # [B, L_test, D_s]
        f_test: jax.Array,  # [B, L_test, D_f]
        valid_lens_ctx: jax.Array | None = None,  # [B]
        strategy: Strategy | None = None,
        num_samples_for_random: int = 0,  # number of samples for Monte Carlo estimation for the random strategy
        batching_for_random: int | None = None,
        # batch the random ll estimation (batch size effectively becomes B*this)
    ):
        if valid_lens_ctx is None:
            B, L_ctx, _ = s_ctx.shape
            valid_lens_ctx = jnp.repeat(L_ctx, B)
        match strategy:
            case None:
                return self._logpdf(s_ctx, f_ctx, s_test, f_test, valid_lens_ctx)
            case "random":
                return self._logpdf_random_estimate(
                    rng,
                    s_ctx,
                    f_ctx,
                    s_test,
                    f_test,
                    valid_lens_ctx,
                    num_samples_for_random,
                    batching_for_random,
                )
            case _:
                idx = strategy.get_batch_permutation(rng, s_ctx, s_test)

        D_s = s_test.shape[-1]
        D_f = f_test.shape[-1]
        idx_s = idx[..., None].repeat(D_s, axis=-1)
        idx_f = idx[..., None].repeat(D_f, axis=-1)
        s_test = jnp.take_along_axis(s_test, idx_s, axis=1)
        f_test = jnp.take_along_axis(f_test, idx_f, axis=1)
        return self._logpdf(s_ctx, f_ctx, s_test, f_test, valid_lens_ctx)
