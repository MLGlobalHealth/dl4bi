import jax.numpy as jnp
from jax import random

from dsp.core import (
    MLP,
    AdditiveScorer,
    DotScorer,
    FixedSinusoidalEmbedding,
    GaussianFourierEmbedding,
    LearnableEmbedding,
    MultiheadAttention,
    MultiheadFastAttention,
    MultiplicativeScorer,
    NeRFEmbedding,
    TransformerDecoder,
    TransformerEncoder,
)


def test_transformer_encoder():
    batch_size, seq_len, embed_dim, feature_dim = 4, 7, 12, 2
    key = random.key(42)
    rng_data, rng_init = random.split(key)
    s = random.normal(rng_data, (batch_size, seq_len, feature_dim))
    valid_lens = jnp.array([2, 4, 6, 3])
    for embedder in [
        FixedSinusoidalEmbedding(embed_dim // feature_dim),
        NeRFEmbedding(embed_dim // feature_dim),
        GaussianFourierEmbedding(embed_dim // 2),
        LearnableEmbedding(post_process=MLP([embed_dim, embed_dim])),
    ]:
        s_e, _ = embedder.init_with_output(rng_init, s)
        for scorer in [AdditiveScorer(), MultiplicativeScorer(), DotScorer()]:
            attn = MultiheadAttention(scorer)
            f_enc, _ = TransformerEncoder(attn).init_with_output(
                rng_init, s_e, valid_lens
            )
            f_dec, _ = TransformerDecoder(attn).init_with_output(
                rng_init, s_e, f_enc, valid_lens, valid_lens
            )
            for name, f in [("encoder", f_enc), ("decoder", f_dec)]:
                assert f_enc.shape == (
                    batch_size,
                    seq_len,
                    embed_dim,
                ), f"Incorrect {name} output shape!"
                assert not jnp.isnan(f).any(), f"{name.title()} returned nans!"
        # test fast version too
        attn = MultiheadFastAttention()
        f_enc, _ = TransformerEncoder(attn).init_with_output(rng_init, s_e, valid_lens)
        f_dec, _ = TransformerDecoder(attn).init_with_output(
            rng_init, s_e, f_enc, valid_lens, valid_lens
        )
        for name, f in [("encoder", f_enc), ("decoder", f_dec)]:
            assert f_enc.shape == (
                batch_size,
                seq_len,
                embed_dim,
            ), f"Incorrect {name} (fast) output shape!"
            assert not jnp.isnan(f).any(), f"{name.title()} (fast) returned nans!"
