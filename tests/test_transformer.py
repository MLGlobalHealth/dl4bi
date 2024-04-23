import jax.numpy as jnp
from jax import random

from dge import (
    FixedSinusoidalEmbedding,
    GaussianFourierEmbedding,
    NeRFEmbedding,
    TransformerEncoder,
)


def test_transformer_encoder():
    batch_size, seq_len, embed_dim, feature_dim = 4, 7, 12, 2
    key = random.key(42)
    rng_data, rng_B, rng_init = random.split(key, 3)
    x = random.normal(rng_data, (batch_size, seq_len, feature_dim))
    B = random.normal(rng_B, (embed_dim, feature_dim))
    valid_lens = jnp.array([2, 4, 6, 3])
    for embed in [
        # TODO(danj): think of best way to collapse multiple dimensions into 1
        # FixedSinusoidalEmbedding(embed_dim),
        # NeRFEmbedding(embed_dim),
        GaussianFourierEmbedding(B),
    ]:
        x, _ = TransformerEncoder(embed).init_with_output(rng_init, x, valid_lens)
        assert x.shape == (
            batch_size,
            seq_len,
            embed_dim,
        ), "Incorrect encoder output shape!"
