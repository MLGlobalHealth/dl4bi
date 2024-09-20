import flax.linen as nn
import flax.linen.initializers as init
import jax
import jax.numpy as jnp
from jax.nn import dot_product_attention
from jax.tree_util import Partial

resid_init = init.normal(0.02 / jnp.sqrt(2 * 12))  # 2 resid for each of 12 layers
Embed = Partial(nn.Embed, dtype=jnp.bfloat16, embedding_init=init.normal(0.02))
Dense = Partial(nn.Dense, dtype=jnp.bfloat16, kernel_init=init.normal(0.02))
LayerNorm = Partial(nn.LayerNorm, dtype=jnp.bfloat16, use_bias=False)
DenseResid = Partial(nn.Dense, dtype=jnp.bfloat16, kernel_init=resid_init)


class MultiheadCausalAttention(nn.Module):
    num_heads: int = 4
    d_model: int = 128

    @nn.compact
    def __call__(self, x: jax.Array):
        (B, L, _), D, H = x.shape, self.d_model, self.num_heads
        qs = Dense(D)(x).reshape(B, L, H, D // H)
        ks = Dense(D)(x).reshape(B, L, H, D // H)
        vs = Dense(D)(x).reshape(B, L, H, D // H)
        ctx = dot_product_attention(qs, ks, vs, is_causal=True, implementation="cudnn")
        return DenseResid(D)(ctx.reshape(B, L, D))


class FFN(nn.Module):
    d_model: int = 128
    p_dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False):
        x = Dense(4 * self.d_model)(x)
        x = nn.gelu(x)
        x = DenseResid(self.d_model)(x)
        return nn.Dropout(self.p_dropout, deterministic=not training)(x)


class Block(nn.Module):
    num_heads: int = 4
    d_model: int = 128
    p_dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False):
        x += MultiheadCausalAttention(self.num_heads, self.d_model)(LayerNorm()(x))
        x += FFN(self.d_model, self.p_dropout)(LayerNorm()(x), training)
        return x


class GPT(nn.Module):
    d_model: int = 768
    num_blks: int = 12
    num_reps: int = 1
    num_heads: int = 12
    num_vocab: int = 50304
    num_context_window: int = 1024
    p_dropout: float = 0.0

    @nn.compact
    def __call__(self, token_ids: jax.Array, training: bool = False):
        embed_tok = Embed(self.num_vocab, self.d_model)
        embed_pos = Embed(self.num_context_window, self.d_model)
        x = embed_tok(token_ids) + embed_pos(jnp.arange(self.num_context_window))
        x = nn.Dropout(self.p_dropout, deterministic=not training)(x)
        for _ in range(self.num_blks):
            blk = Block(self.num_heads, self.d_model, self.p_dropout)
            for _ in range(self.num_reps):
                x = blk(x, training)
        x = LayerNorm()(x)
        return embed_tok.attend(x)
