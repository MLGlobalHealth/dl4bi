import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, List, Tuple


class PatchEmbedding(nn.Module):
    patch_size: int
    embed_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        h, w = H // self.patch_size, W // self.patch_size
        # reshape into patches
        x = x.reshape(B, h, self.patch_size, w, self.patch_size, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, h * w, self.patch_size * self.patch_size * C)
        # linear projection + pos embedding
        x = nn.Dense(self.embed_dim)(x)
        pos_emb = self.param(
            "pos_emb", nn.initializers.normal(1.0), (1, h * w, self.embed_dim)
        )
        return x + pos_emb


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_dim: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # self-attention
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            features=self.embed_dim, num_heads=self.num_heads, dropout_rate=self.dropout
        )(y, y, deterministic=deterministic)
        x = x + y
        # MLP
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        y = nn.Dense(self.embed_dim)(y)
        return x + y


class PseudoSkip(nn.Module):
    embed_dim: int
    num_tokens: int
    num_heads: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, context: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # context: (B, N, D)
        # Initialize K learnable skip tokens
        skip_tokens = self.param(
            "skip_tokens",
            nn.initializers.normal(1.0),
            (1, self.num_tokens, self.embed_dim),
        )
        # broadcast to batch
        skips = jnp.tile(skip_tokens, (context.shape[0], 1, 1))
        # cross-attention: skips attend to context
        skips = nn.LayerNorm()(skips)
        skips = nn.MultiHeadDotProductAttention(
            features=self.embed_dim, num_heads=self.num_heads, dropout_rate=self.dropout
        )(skips, context, deterministic=deterministic)
        return skips  # (B, K, D)


class UNetTransformer(nn.Module):
    img_size: int
    patch_size: int
    embed_dim: int
    depth: List[int]  # number of blocks at each stage
    num_heads: List[int]
    mlp_dim: List[int]
    skip_tokens: List[int]
    num_classes: int = 3  # channels in output

    def setup(self):
        # encoder
        self.patch_embed = PatchEmbedding(self.patch_size, self.embed_dim)
        self.enc_blocks = []
        self.skip_modules = []
        for i, d in enumerate(self.depth):
            blocks = [
                TransformerBlock(self.embed_dim, self.num_heads[i], self.mlp_dim[i])
                for _ in range(d)
            ]
            self.enc_blocks.append(blocks)
            self.skip_modules.append(
                PseudoSkip(self.embed_dim, self.skip_tokens[i], self.num_heads[i])
            )
        # bottleneck
        self.bottleneck = [
            TransformerBlock(self.embed_dim, self.num_heads[-1], self.mlp_dim[-1])
        ]
        # decoder (mirror of encoder)
        self.dec_blocks = []
        self.cross_attn = []
        for i, d in enumerate(reversed(self.depth)):
            blocks = [
                TransformerBlock(
                    self.embed_dim, self.num_heads[-(i + 1)], self.mlp_dim[-(i + 1)]
                )
                for _ in range(d)
            ]
            self.dec_blocks.append(blocks)
            self.cross_attn.append(
                nn.MultiHeadDotProductAttention(
                    features=self.embed_dim,
                    num_heads=self.num_heads[-(i + 1)],
                    dropout_rate=0.1,
                )
            )
        # output projection
        self.to_image = nn.Sequential(
            [
                nn.LayerNorm(),
                nn.Dense(self.patch_size * self.patch_size * self.num_classes),
            ]
        )

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        # x: (B, H, W, C)
        B = x.shape[0]
        # ---- Encoder + pseudo-skips ----
        tokens = self.patch_embed(x)
        skip_list: List[jnp.ndarray] = []
        for blocks, skip_mod in zip(self.enc_blocks, self.skip_modules):
            for blk in blocks:
                tokens = blk(tokens, deterministic=not train)
            skips = skip_mod(tokens, deterministic=not train)
            skip_list.append(skips)
        # ---- Bottleneck ----
        for blk in self.bottleneck:
            tokens = blk(tokens, deterministic=not train)
        # ---- Decoder ----
        # initialize decoder tokens as current tokens
        dec_tokens = tokens
        for skips, blocks, cross in zip(
            skip_list[::-1], self.dec_blocks, self.cross_attn
        ):
            # cross-attend to skip tokens
            dec_tokens = cross(dec_tokens, skips)
            # transformer blocks
            for blk in blocks:
                dec_tokens = blk(dec_tokens, deterministic=not train)
        # ---- Reconstruct image ----
        # project tokens back to patches
        patches = self.to_image(dec_tokens)  # (B, N, P*P*C)
        # reshape
        h, w = self.img_size // self.patch_size, self.img_size // self.patch_size
        patches = patches.reshape(
            B, h, w, self.patch_size, self.patch_size, self.num_classes
        )
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        imgs = patches.reshape(B, self.img_size, self.img_size, self.num_classes)
        return imgs
