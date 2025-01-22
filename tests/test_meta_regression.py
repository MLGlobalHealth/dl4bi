import jax
import jax.numpy as jnp
import optax
from jax import jit, random

from dl4bi.core import (
    Attention,
    DeepKernelAttention,
    FastAttention,
    FusedAttention,
    KRBlock,
    MultiHeadAttention,
    kNN,
)
from dl4bi.meta_regression import (
    ANP,
    BANP,
    BNP,
    CANP,
    CNP,
    DSKR,
    NP,
    TNPD,
    TNPKR,
    TNPND,
    ConvCNP,
    ScanTNPKR,
)
from dl4bi.meta_regression import (
    train_utils as tu,
)


def test_models():
    B, L = 4, 10
    key = random.key(42)
    rng_data, rng_params, rng_dropout, rng_extra = random.split(key, 4)
    s = jnp.linspace(0, 1.0, L)
    s = jnp.repeat(s[None, :, None], B, axis=0)  # [B, S, D_s=1]
    valid_lens = jnp.array([2, 4, 9, 3])
    f = random.normal(rng_data, s.shape)
    for model in [
        NP,
        CNP,
        BNP,
        ANP,
        CANP,
        BANP,
        lambda: DSKR(kNN(k=L)),
        TNPD,
        TNPND,
        TNPKR,
        ScanTNPKR,
        ConvCNP,
    ]:
        m = model()
        output, params = m.init_with_output(
            {"params": rng_params, "dropout": rng_dropout, "extra": rng_extra},
            s_ctx=s,
            f_ctx=f,
            s_test=s,
            valid_lens_ctx=valid_lens,
            valid_lens_test=valid_lens,
            training=True,
        )
        if hasattr(m, "n_z"):  # latent model
            output, _ = output  # throw away latent zs
        if hasattr(m, "num_samples"):  # bootstrapped model
            output, _ = output  # throw away base output
        f_mu, f_std = output
        K = f_mu.shape[0] // f.shape[0]
        assert f_mu.shape == (B * K, L, 1)


def test_tnp_kr_fast_scale():
    B, D_f, L_init, L_ctx, L_test = 1, 1, 3, 500000, 50000
    rng = random.key(42)
    rng_ctx, rng_init, rng_model = random.split(rng, 3)
    s_init = jnp.linspace(0, 1.0, L_init)[None, :, None]  # [1, L_init, 1]
    s_ctx = jnp.linspace(0, 1.0, L_ctx)[None, :, None]  # [1, L_ctx, 1]
    s_test = jnp.linspace(0, 1.0, L_test)[None, :, None]  # [1, L_test, 1]
    f_init = random.normal(rng_init, (B, L_init, D_f))
    f_ctx = random.normal(rng_ctx, (B, L_ctx, D_f))
    m = TNPKR(blk=KRBlock(MultiHeadAttention(FastAttention())))
    params = m.init(rng_init, s_init, f_init, s_init)
    jit_m = jit(lambda *args: m.apply(params, *args))
    jit_m(s_init, f_init, s_init)  # dummy run to compile
    # to view results: tensorboard --logdir /tmp/tensorboard/
    with jax.profiler.trace("/tmp/tensorboard"):
        f_mu, f_std, *_ = jit_m(s_ctx, f_ctx, s_test)
    assert jnp.isfinite(f_mu).all(), "Non-finite values produced!"
    assert jnp.isfinite(f_std).all(), "Non-finite values produced!"


def test_context_data_leaks():
    B, L, N = 1, 256, 128
    key = random.key(42)
    rng_data, rng_params, rng_dropout, rng_extra = random.split(key, 4)
    s = jnp.linspace(-2.0, 2.0, L)
    s = jnp.repeat(s[None, :, None], B, axis=0)  # [B, L, S=1]
    valid_lens_ctx = jnp.array([N] * B, dtype=jnp.int32)
    valid_lens_test = jnp.array([L] * B, dtype=jnp.int32)
    f = random.normal(rng_data, s.shape)
    # set second half to large value (different from using half the array because of attn)
    s2 = s.at[:, N:, :].set(jnp.full((B, L - N, 1), 1000))
    f2 = f.at[:, N:, :].set(jnp.full((B, L - N, 1), 1000))
    for model in [
        NP,
        CNP,
        BNP,
        ANP,
        CANP,
        BANP,
        ConvCNP,
        TNPD,
        TNPND,
        lambda: TNPKR(blk=KRBlock(MultiHeadAttention(Attention()))),
        lambda: TNPKR(blk=KRBlock(MultiHeadAttention(FusedAttention()))),
        lambda: TNPKR(blk=KRBlock(MultiHeadAttention(FastAttention()))),
        lambda: TNPKR(blk=KRBlock(DeepKernelAttention())),
        ScanTNPKR,
        lambda: DSKR(kNN(k=32), num_blks=1),
    ]:
        print(model)
        m = model()
        output, params = m.init_with_output(
            {"params": rng_params, "dropout": rng_dropout, "extra": rng_extra},
            s_ctx=s,
            f_ctx=f,
            s_test=s,
            valid_lens_ctx=valid_lens_ctx,
            valid_lens_test=valid_lens_test,
            bucket_size=2,  # used by DSKR for numerical stability
        )
        output_half = m.apply(
            params,
            s_ctx=s2,
            f_ctx=f2,
            s_test=s,
            valid_lens_ctx=valid_lens_ctx,
            valid_lens_test=valid_lens_test,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
            bucket_size=2,  # used by DSKR for numerical stability
        )
        if hasattr(model, "n_z"):  # latent model
            output, _ = output  # throw away latent zs
            output_half, _ = output_half
        if hasattr(m, "num_samples"):  # bootstrapped model
            output, _ = output  # throw away base output
            output_half, _ = output_half
        f_mu, f_std = output
        f_mu_half, f_std_half = output_half
        print(jnp.sort(jnp.abs(f_mu - f_mu_half).flatten(), descending=True)[:5])
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
    for model in [
        NP,
        CNP,
        BNP,
        ANP,
        CANP,
        BANP,
        lambda: DSKR(kNN(k=L)),
        TNPD,
        TNPND,
        TNPKR,
        ScanTNPKR,
        ConvCNP,
    ]:
        print(model)
        m = model()
        train_step, _ = tu.select_steps(m)
        kwargs = m.init(
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
            apply_fn=m.apply,
            params=params,
            tx=optax.yogi(learning_rate_fn),
            kwargs=kwargs,
        )
        _, loss_1 = train_step(rng_step, state, batch_1)
        _, loss_2 = train_step(rng_step, state, batch_2)
        assert jnp.not_equal(loss_1, loss_2)


def test_sample():
    rng = random.key(42)
    rng_params, rng_dropout, rng_extra, rng_sample = random.split(rng, 4)
    s_ctx = jnp.linspace(0, 0.90, 90)[:, None]  # [L_ctx, 1]
    s_test = jnp.linspace(0.90, 1.0, 10)[:, None]  # [L_test, 1]
    f_ctx = s_ctx  # [L_ctx, 1] : just a line
    for model in [
        NP,
        CNP,
        ANP,
        CANP,
        lambda: DSKR(kNN(k=10)),
        TNPD,
        TNPKR,
        ScanTNPKR,
        ConvCNP,
    ]:
        print(model)
        m = model()
        kwargs = m.init(
            {"params": rng_params, "dropout": rng_dropout, "extra": rng_extra},
            s_ctx[None, ...],
            f_ctx[None, ...],
            s_test[None, ...],
            valid_lens_ctx=jnp.array(s_ctx.shape[0])[None, ...],
        )
        params = kwargs.pop("params")
        learning_rate_fn = tu.cosine_annealing_lr()
        state = tu.TrainState.create(
            apply_fn=m.apply,
            params=params,
            tx=optax.yogi(learning_rate_fn),
            kwargs=kwargs,
        )
        tu.sample(rng_sample, state, s_ctx, f_ctx, s_test, batch_size=32)
