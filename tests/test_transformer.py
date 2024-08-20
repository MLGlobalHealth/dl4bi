import jax.numpy as jnp
from jax import random

from dsp.core import (
    MLP,
    AdditiveScorer,
    DotScorer,
    FixedSinusoidalEmbedding,
    GaussianFourierEmbedding,
    MultiheadAttention,
    MultiheadFastAttention,
    MultiplicativeScorer,
    NeRFEmbedding,
    TransformerDecoder,
    TransformerDecoderBlock,
    TransformerEncoder,
    TransformerEncoderBlock,
)


def test_transformer_encoder():
    batch_size, seq_len, embed_dim, feature_dim = 4, 7, 64, 2
    key = random.key(42)
    rng_data, rng_init = random.split(key)
    s = random.normal(rng_data, (batch_size, seq_len, feature_dim))
    valid_lens = jnp.array([2, 4, 6, 3])
    for embedder in [
        FixedSinusoidalEmbedding(embed_dim // feature_dim),
        NeRFEmbedding(embed_dim // feature_dim),
        GaussianFourierEmbedding(embed_dim),
        MLP([embed_dim, embed_dim]),
    ]:
        s_e, _ = embedder.init_with_output(rng_init, s)
        for scorer in [AdditiveScorer(), MultiplicativeScorer(), DotScorer()]:
            attn = MultiheadAttention(scorer=scorer)
            enc_blk = TransformerEncoderBlock(attn)
            f_enc, _ = TransformerEncoder(blk=enc_blk).init_with_output(
                rng_init, s_e, valid_lens
            )
            dec_blk = TransformerDecoderBlock(attn)
            f_dec, _ = TransformerDecoder(blk=dec_blk).init_with_output(
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
        enc_blk = TransformerEncoderBlock(attn)
        dec_blk = TransformerDecoderBlock(attn)
        f_enc, _ = TransformerEncoder(blk=enc_blk).init_with_output(
            rng_init, s_e, valid_lens
        )
        f_dec, _ = TransformerDecoder(blk=dec_blk).init_with_output(
            rng_init, s_e, f_enc, valid_lens, valid_lens
        )
        for name, f in [("encoder", f_enc), ("decoder", f_dec)]:
            assert f_enc.shape == (
                batch_size,
                seq_len,
                embed_dim,
            ), f"Incorrect {name} (fast) output shape!"
            assert not jnp.isnan(f).any(), f"{name.title()} (fast) returned nans!"
