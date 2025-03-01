import jax.numpy as jnp
from jax import random

from dl4bi.core.attention import (
    MLP,
    Attention,
    FastAttention,
    MultiHeadAttention,
)
from dl4bi.core.embed import (
    FixedSinusoidalEmbedding,
    GaussianFourierEmbedding,
    NeRFEmbedding,
)
from dl4bi.core.transformer import (
    TransformerDecoder,
    TransformerDecoderBlock,
    TransformerEncoder,
    TransformerEncoderBlock,
)
from dl4bi.core.utils import mask_from_valid_lens


def test_transformer():
    B, L, H, E, D = 4, 7, 4, 64, 2
    key = random.key(42)
    rng_data, rng_bias, rng_init = random.split(key, 3)
    s = random.normal(rng_data, (B, L, D))
    bias = random.normal(rng_bias, (B, H, L, L))
    valid_lens = jnp.array([2, 4, 6, 3])
    mask = mask_from_valid_lens(L, valid_lens)
    for embedder in [
        FixedSinusoidalEmbedding(E // D),
        NeRFEmbedding(E // D),
        GaussianFourierEmbedding(E),
        MLP([E, E]),
    ]:
        s_e, _ = embedder.init_with_output(rng_init, s)
        attn = MultiHeadAttention(Attention())
        enc_blk = TransformerEncoderBlock(attn)
        f_enc, _ = TransformerEncoder(blk=enc_blk).init_with_output(
            rng_init, s_e, mask, bias=bias
        )
        dec_blk = TransformerDecoderBlock(attn)
        f_dec, _ = TransformerDecoder(blk=dec_blk).init_with_output(
            rng_init,
            s_e,
            f_enc,
            mask,
            mask,
            qq_kwargs={"bias": bias},
            qk_kwargs={"bias": bias},
        )
        for name, f in [("encoder", f_enc), ("decoder", f_dec)]:
            assert f_enc.shape == (B, L, E), f"Incorrect {name} output shape!"
            assert not jnp.isnan(f).any(), f"{name.title()} returned nans!"
        # test fast version too
        mh_attn = MultiHeadAttention(attn=FastAttention())
        enc_blk = TransformerEncoderBlock(mh_attn)
        dec_blk = TransformerDecoderBlock(mh_attn)
        f_enc, _ = TransformerEncoder(blk=enc_blk).init_with_output(rng_init, s_e, mask)
        f_dec, _ = TransformerDecoder(blk=dec_blk).init_with_output(
            rng_init, s_e, f_enc, mask
        )
        for name, f in [("encoder", f_enc), ("decoder", f_dec)]:
            assert f_enc.shape == (B, L, E), f"Incorrect {name} (fast) output shape!"
            assert not jnp.isnan(f).any(), f"{name.title()} (fast) returned nans!"
