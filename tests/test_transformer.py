import jax.numpy as jnp
from jax import random

from dge import (
    MLP,
    AdditiveScorer,
    DotScorer,
    FixedSinusoidalEmbedding,
    GaussianFourierEmbedding,
    LearnableEmbedding,
    MultiplicativeScorer,
    NeRFEmbedding,
    TransformerEncoder,
)


def test_transformer_encoder():
    batch_size, seq_len, embed_dim, feature_dim = 4, 7, 12, 2
    key = random.key(42)
    rng_data, rng_B, rng_init = random.split(key, 3)
    s = random.normal(rng_data, (batch_size, seq_len, feature_dim))
    B = random.normal(rng_B, (embed_dim // 2, feature_dim))
    valid_lens = jnp.array([2, 4, 6, 3])
    for embedder in [
        FixedSinusoidalEmbedding(embed_dim // feature_dim),
        NeRFEmbedding(embed_dim // feature_dim),
        GaussianFourierEmbedding(B),
        LearnableEmbedding(
            FixedSinusoidalEmbedding(embed_dim // feature_dim),
            MLP([embed_dim, embed_dim]),
        ),
    ]:
        s_e, _ = embedder.init_with_output(rng_init, s)
        for scorer in [AdditiveScorer(), MultiplicativeScorer(), DotScorer()]:
            f, _ = TransformerEncoder(scorer).init_with_output(
                rng_init, s_e, valid_lens
            )
            assert f.shape == (
                batch_size,
                seq_len,
                embed_dim,
            ), "Incorrect encoder output shape!"
