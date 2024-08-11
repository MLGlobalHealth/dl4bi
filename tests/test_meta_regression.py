import os

import jax
import jax.numpy as jnp
import optax
from jax import jit, random

from dsp.meta_regression import (
    ANP,
    BANP,
    BNP,
    CANP,
    CNP,
    DKR,
    NP,
    TNPD,
    TNPDS,
    TNPKR,
    TNPND,
    ConvCNP,
)
from dsp.meta_regression import (
    train_utils as tu,
)

# https://jax.readthedocs.io/en/latest/gpu_performance_tips.html#code-generation-flags
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


def test_models():
    B, L = 4, 10
    key = random.key(42)
    rng_data, rng_params, rng_dropout, rng_extra = random.split(key, 4)
    s = jnp.linspace(0, 1.0, L)
    s = jnp.repeat(s[None, :, None], B, axis=0)  # [B, S, D_s=1]
    valid_lens = jnp.array([2, 4, 9, 3])
    f = random.normal(rng_data, s.shape)
    for np in [NP, CNP, BNP, ANP, CANP, BANP, DKR, TNPD, TNPDS, TNPND, TNPKR, ConvCNP]:
        m = np()
        (f_mu, f_std, *_), params = m.init_with_output(
            {"params": rng_params, "dropout": rng_dropout, "extra": rng_extra},
            s_ctx=s,
            f_ctx=f,
            s_test=s,
            valid_lens_ctx=valid_lens,
            valid_lens_test=valid_lens,
            training=True,
        )
        K = f_mu.shape[0] // f.shape[0]
        assert f_mu.shape == (B * K, L, 1)


def test_tnp_kr_scale():
    B, D_f, L_init, L_ctx, L_test = 1, 1, 3, 1000000, 100000
    rng = random.key(42)
    rng_ctx, rng_init, rng_model = random.split(rng, 3)
    s_init = jnp.linspace(0, 1.0, L_init)[None, :, None]  # [1, L_init, 1]
    s_ctx = jnp.linspace(0, 1.0, L_ctx)[None, :, None]  # [1, L_ctx, 1]
    s_test = jnp.linspace(0, 1.0, L_test)[None, :, None]  # [1, L_test, 1]
    f_init = random.normal(rng_init, (B, L_init, D_f))
    f_ctx = random.normal(rng_ctx, (B, L_ctx, D_f))
    m = TNPKR()
    params = m.init(rng_init, s_init, f_init, s_init)
    jit_m = jit(lambda *args: m.apply(params, *args))
    jit_m(s_init, f_init, s_init)  # dummy run to compile
    # to view results: tensorboard --logdir /tmp/tensorboard/
    with jax.profiler.trace("/tmp/tensorboard"):
        f_mu, f_std, *_ = jit_m(s_ctx, f_ctx, s_test)
    assert jnp.isfinite(f_mu).all(), "Non-finite values produced!"
    assert jnp.isfinite(f_std).all(), "Non-finite values produced!"


def test_context_data_leaks():
    B, L, N = 4, 10, 5
    key = random.key(42)
    rng_data, rng_params, rng_dropout, rng_extra = random.split(key, 4)
    s = jnp.linspace(0, 1.0, L)
    s = jnp.repeat(s[None, :, None], B, axis=0)  # [B, S, D_s=1]
    valid_lens_ctx = jnp.array([N] * B)
    valid_lens_test = jnp.array([L] * B)
    f = 10 * random.normal(rng_data, s.shape)
    # set second half to 0s (different from using half the array because of attn)
    s2 = s.at[:, N:, :].set(jnp.zeros((B, L - N, 1)))
    f2 = f.at[:, N:, :].set(jnp.zeros((B, L - N, 1)))
    for np in [NP, CNP, BNP, ANP, CANP, BANP, DKR, TNPD, TNPDS, TNPND, TNPKR, ConvCNP]:
        print(np)
        m = np()
        (f_mu, f_std, *_), params = m.init_with_output(
            {"params": rng_params, "dropout": rng_dropout, "extra": rng_extra},
            s_ctx=s,
            f_ctx=f,
            s_test=s,
            valid_lens_ctx=valid_lens_ctx,
            valid_lens_test=valid_lens_test,
        )
        f_mu_half, f_std_half, *_ = m.apply(
            params,
            s_ctx=s2,
            f_ctx=f2,
            s_test=s,
            valid_lens_ctx=valid_lens_ctx,
            valid_lens_test=valid_lens_test,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
        )
        assert jnp.allclose(f_mu, f_mu_half)
        assert jnp.allclose(f_std, f_std_half)


def test_train_step_loss():
    B, L, N = 4, 10, 5
    key = random.key(42)
    rng_data, rng_params, rng_dropout, rng_extra, rng_step = random.split(key, 5)
    s = jnp.linspace(0, 1.0, L)
    s = jnp.repeat(s[None, :, None], B, axis=0)  # [B, S, D_s=1]
    valid_lens_ctx = jnp.array([N] * B)
    valid_lens_test_1 = jnp.array([L] * B)
    valid_lens_test_2 = jnp.array([L - 1] * B)
    f = 10 * random.normal(rng_data, s.shape)
    batch_1 = (s, f, valid_lens_ctx, s, f, valid_lens_test_1)
    batch_2 = (s, f, valid_lens_ctx, s, f, valid_lens_test_2)
    for np in [NP, CNP, BNP, ANP, CANP, BANP, DKR, TNPD, TNPDS, TNPND, TNPKR, ConvCNP]:
        print(np)
        model = np()
        train_step = tu.vanilla_train_step
        if isinstance(model, (NP, ANP)):
            train_step = tu.npf_elbo_train_step
        elif isinstance(model, (BNP, BANP)):
            train_step = tu.bootstrap_train_step
        elif isinstance(model, (TNPND,)):
            train_step = tu.tril_cov_train_step
        kwargs = model.init(
            {"params": rng_params, "dropout": rng_dropout, "extra": rng_extra},
            s_ctx=s,
            f_ctx=f,
            s_test=s,
            valid_lens_ctx=valid_lens_ctx,
            valid_lens_test=valid_lens_test_1,
        )
        params = kwargs.pop("params")
        learning_rate_fn = tu.cosine_annealing_lr()
        state = tu.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.yogi(learning_rate_fn),
            kwargs=kwargs,
        )
        _, loss_1 = train_step(rng_step, state, batch_1)
        _, loss_2 = train_step(rng_step, state, batch_2)
        assert jnp.not_equal(loss_1, loss_2)
