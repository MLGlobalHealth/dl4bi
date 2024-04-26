import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random

from .attention import Attention
from .embed import FixedSinusoidalEmbedding
from .mlp import MLP
from .transformer import TransformerEncoder


# TODO(danj): embed target values as well? or target + loc values together?
# TODO(danj): add valid_lens for ragged input
class AttentiveNeuralProcess(nn.Module):
    location_embedder: nn.Module = FixedSinusoidalEmbedding()
    local_s_and_f_embedder: nn.Module = TransformerEncoder()
    local_attention: nn.Module = Attention()
    global_s_and_f_embedder: nn.Module = TransformerEncoder()
    global_mu_z_decoder: nn.Module = MLP([128, 128])
    global_log_var_z_decoder: nn.Module = MLP([128, 128])
    mu_f_decoder: nn.Module = MLP([128 * 3, 128, 1])
    log_var_f_decoder: nn.Module = MLP([128 * 3, 128, 1])

    @nn.compact
    def __call__(
        self,
        key: jax.Array,
        s_ctx: jax.Array,  # [B, S_ctx, D_S]
        f_ctx: jax.Array,  # [B, S_ctx, D_F]
        s_test: jax.Array,  # [B, S_test, D_S]
        training: bool = False,
    ):
        qs = self.location_embedder(s_test)
        ks = self.location_embedder(s_ctx)
        if s_ctx.ndim > f_ctx.ndim:
            f_ctx = f_ctx[..., None]
        s_and_f = jnp.concatenate([s_ctx, f_ctx], -1)
        # local ("deterministic") path
        local_vs = self.local_s_and_f_embedder(s_and_f, training)
        local_ctx, _ = self.local_attention(qs, ks, local_vs, training)
        # global ("latent") path
        global_vs = self.global_s_and_f_embedder(s_and_f, training)
        global_vs_mean = global_vs.mean()
        mu_z = self.global_mu_z_decoder(global_vs_mean, training)
        log_var_z = self.global_log_var_z_decoder(global_vs_mean, training)
        global_zs = mu_z + jnp.exp(log_var_z) * random.normal(key, mu_z.shape)
        # decoding context to (mu_f, log_var_f) for every test location
        print(qs.shape, local_ctx.shape, global_zs.shape)
        global_zs = jnp.repeat(global_zs[:, jnp.newaxis, :], qs.shape[1], axis=1)
        context = jnp.concatenate([qs, local_ctx, global_zs], -1)
        mu_f = self.mu_f_decoder(context)
        log_var_f = self.log_var_f_decoder(context)
        return global_zs, mu_f, log_var_f
