import jax
import jax.numpy as jnp
import optax
from jax import jit, random

from dl4bi.core.attention import (
    Attention,
    BiasedScanAttention,
    DeepKernelAttention,
    FastAttention,
    MultiHeadAttention,
)
from dl4bi.core.bias import Bias
from dl4bi.core.sim import l2_dist
from dl4bi.core.train import TrainState, cosine_annealing_lr
from dl4bi.core.transformer import KRBlock
from dl4bi.core.utils import mask_from_valid_lens
from dl4bi.meta_learning import (
    ANP,
    BSATNP,
    BTNP,
    CANP,
    CNP,
    NP,
    SGNP,
    TETNP,
    TNPD,
    ConvCNP,
)
from dl4bi.meta_learning.data.spatial import SpatialData
from dl4bi.meta_learning.utils import so3_rotate


def test_models():
    B, L = 4, 128
    key = random.key(42)
    rng_data, rng_params, rng_dropout, rng_extra = random.split(key, 4)
    s = jnp.linspace(0, 1.0, L)
    s = jnp.repeat(s[None, :, None], B, axis=0)  # [B, S, D_s=1]
    valid_lens = jnp.array([22, 44, 97, 32])
    mask_ctx = mask_from_valid_lens(L, valid_lens)
    f = random.normal(rng_data, s.shape)
    bias = Bias.build_rbf_network_bias()
    scanned_kr_block = KRBlock(
        MultiHeadAttention(BiasedScanAttention(bias={"s": bias}))
    )
    for model in [
        NP,
        CNP,
        ANP,
        CANP,
        ConvCNP,
        TNPD,
        TETNP,
        BTNP,
        lambda: BTNP(blk=KRBlock(MultiHeadAttention(Attention()))),
        lambda: BTNP(blk=KRBlock(MultiHeadAttention(FastAttention()))),
        lambda: BTNP(blk=KRBlock(DeepKernelAttention())),
        lambda: BTNP(s_sim=l2_dist, s_bias=Bias.build_rbf_network_bias()),
        lambda: BSATNP(blk=scanned_kr_block),
        lambda: SGNP(s_sim=l2_dist, s_bias=Bias.build_rbf_network_bias()),
    ]:
        m = model()
        print(m.__class__)
        output, params = m.init_with_output(
            {"params": rng_params, "dropout": rng_dropout, "extra": rng_extra},
            s_ctx=s,
            f_ctx=f,
            s_test=s,
            mask_ctx=mask_ctx,
            training=True,
        )
        if isinstance(output, tuple):
            output, _ = output  # drop latent distribution
        assert jnp.isfinite(output.mu).all()
        assert jnp.isfinite(output.std).all()
        assert output.mu.shape == (B, L, 1)
        assert output.std.shape == (B, L, 1)


def test_tnp_kr_fast_scale():
    B, D_f, L_init, L_ctx, L_test = 1, 1, 3, 500000, 50000
    rng = random.key(42)
    rng_ctx, rng_init, rng_model = random.split(rng, 3)
    s_init = jnp.linspace(0, 1.0, L_init)[None, :, None]  # [1, L_init, 1]
    s_ctx = jnp.linspace(0, 1.0, L_ctx)[None, :, None]  # [1, L_ctx, 1]
    s_test = jnp.linspace(0, 1.0, L_test)[None, :, None]  # [1, L_test, 1]
    f_init = random.normal(rng_init, (B, L_init, D_f))
    f_ctx = random.normal(rng_ctx, (B, L_ctx, D_f))
    m = BTNP(blk=KRBlock(MultiHeadAttention(FastAttention())))
    params = m.init(rng_init, s_ctx=s_init, f_ctx=f_init, s_test=s_init)
    jit_m = jit(lambda **kwargs: m.apply(params, **kwargs))
    jit_m(s_ctx=s_init, f_ctx=f_init, s_test=s_init)  # dummy run to compile
    # to view results: tensorboard --logdir /tmp/tensorboard/
    with jax.profiler.trace("/tmp/tensorboard"):
        output = jit_m(s_ctx=s_ctx, f_ctx=f_ctx, s_test=s_test)
    assert jnp.isfinite(output.mu).all(), "Non-finite values produced!"
    assert jnp.isfinite(output.std).all(), "Non-finite values produced!"


def test_context_data_leaks():
    B, L, V = 4, 128, 64
    rng = random.key(42)
    rng_s, rng_f, rng_params, rng_dropout, rng_extra = random.split(rng, 5)
    mask_ctx = jnp.concat(
        [jnp.ones((B, V), dtype=bool), jnp.zeros((B, L - V), dtype=bool)], axis=1
    )
    s = 4 * (random.uniform(rng_s, (B, L, 1)) - 0.5)  # [0, 1] -> [-0.5, 0.5] -> [-2, 2]
    f = random.normal(rng_f, s.shape)
    # set second half to large value (different from using half the array because of attn)
    s2 = s.at[:, V:, :].set(jnp.full((B, L - V, 1), 10000))
    f2 = f.at[:, V:, :].set(jnp.full((B, L - V, 1), 10000))
    bias = Bias.build_rbf_network_bias()
    scanned_kr_block = KRBlock(
        MultiHeadAttention(BiasedScanAttention(bias={"s": bias}))
    )
    for model in [
        NP,
        CNP,
        ANP,
        CANP,
        ConvCNP,
        TETNP,
        TNPD,
        lambda: BTNP(blk=KRBlock(MultiHeadAttention(Attention()))),
        lambda: BTNP(blk=KRBlock(MultiHeadAttention(FastAttention()))),
        lambda: BTNP(blk=KRBlock(DeepKernelAttention())),
        lambda: BSATNP(blk=scanned_kr_block),
        lambda: BTNP(s_sim=l2_dist, s_bias=Bias.build_rbf_network_bias()),
        lambda: SGNP(s_sim=l2_dist, s_bias=Bias.build_rbf_network_bias()),
    ]:
        m = model()
        print(m.__class__)
        params = m.init(
            {"params": rng_params, "dropout": rng_dropout, "extra": rng_extra},
            s_ctx=s,
            f_ctx=f,
            s_test=s,
            mask_ctx=mask_ctx,
        )
        jit_m = jit(m.apply, static_argnames=("bucket_size",))
        output = jit_m(
            params,
            s_ctx=s,
            f_ctx=f,
            s_test=s,
            mask_ctx=mask_ctx,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
            # bucket_size=2,  # used by SGNP for numerical stability
        )
        # to view results: tensorboard --logdir /tmp/tensorboard/
        # with jax.profiler.trace("/tmp/tensorboard"):
        output_half = jit_m(
            params,
            s_ctx=s2,
            f_ctx=f2,
            s_test=s,
            mask_ctx=mask_ctx,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
            # bucket_size=2,  # used by SGNP for numerical stability
        )
        if isinstance(output, tuple):
            output, _ = output  # drop latent distribution
            output_half, _ = output_half
        f_mu, f_std = output.mu, output.std
        f_mu_half, f_std_half = output_half.mu, output_half.std
        print(
            "largest gaps:",
            jnp.sort(jnp.abs(f_mu - f_mu_half).flatten(), descending=True)[:5],
        )
        # jax.ops.segment_sum depends on order of addition and can
        # introduce small numerical variations
        if isinstance(m, SGNP):
            assert jnp.allclose(f_mu, f_mu_half, atol=0.003)
            assert jnp.allclose(f_std, f_std_half, atol=0.003)
        else:
            assert jnp.allclose(f_mu, f_mu_half)
            assert jnp.allclose(f_std, f_std_half)


def test_train_step_loss():
    B, L, N = 4, 64, 32
    rng = random.key(42)
    rng_f, rng_b1, rng_b2, rng = random.split(rng, 4)
    s = jnp.linspace(0, 1.0, L)
    s = jnp.repeat(s[None, :, None], B, axis=0)  # [B, S, D_s=1]
    f = 10 * random.normal(rng_f, s.shape)
    d = SpatialData(x=None, s=s, f=f)
    batch_1 = d.batch(rng_b1, 1, N, L, True)
    batch_2 = d.batch(rng_b2, 1, N, L, True)
    rng_init, rng_drop, rng_extra, rng_step = random.split(rng, 4)
    bias = Bias.build_rbf_network_bias()
    scanned_kr_block = KRBlock(
        MultiHeadAttention(BiasedScanAttention(bias={"s": bias}))
    )
    for model in [
        NP,
        CNP,
        ANP,
        CANP,
        TNPD,
        TETNP,
        BTNP,
        ConvCNP,
        lambda: BTNP(s_sim=l2_dist, s_bias=Bias.build_rbf_network_bias()),
        lambda: BSATNP(blk=scanned_kr_block),
        lambda: SGNP(s_sim=l2_dist, s_bias=Bias.build_rbf_network_bias()),
    ]:
        m = model()
        print(m.__class__)
        kwargs = m.init(
            {"params": rng_init, "dropout": rng_drop, "extra": rng_extra},
            **batch_1,
        )
        params = kwargs.pop("params")
        learning_rate_fn = cosine_annealing_lr()
        state = TrainState.create(
            apply_fn=m.apply,
            params=params,
            tx=optax.yogi(learning_rate_fn),
            kwargs=kwargs,
        )
        _, loss_1 = m.train_step(rng_step, state, batch_1)
        _, loss_2 = m.train_step(rng_step, state, batch_2)
        assert jnp.not_equal(loss_1, loss_2)


def test_rotations():
    s = jnp.array([[0, 0]])

    # 90 north, 20 ccw tilt should give -90, 70
    assert jnp.allclose(so3_rotate(s, 90, 0, 20), jnp.array([[-90, 70]]))

    assert jnp.allclose(so3_rotate(s, 0, 0, 0), jnp.array(s))
    assert jnp.allclose(so3_rotate(s, 0, 90, 0), jnp.array([[90, 0]]))
    assert jnp.allclose(so3_rotate(s, 25, 0, 0), jnp.array([[0, 25]]))
