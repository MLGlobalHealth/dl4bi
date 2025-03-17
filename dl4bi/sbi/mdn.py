from typing import Sequence

import flax.linen as nn
import jax

from ..core.mlp import MLP
from ..core.model_output import MDNOutput


class MLPMDN(nn.Module):
    k: int = 5
    num_hidden: int = 128
    num_layers: int = 3

    @nn.compact
    def __call__(self, x: jax.Array, **kwargs):
        params = MLP([self.num_hidden] * self.num_layers + [3 * self.k])(x)
        return MDNOutput.from_activations(params)


class MDN(nn.Module):
    embed: nn.Module
    est_pi: nn.Module
    est_mu: nn.Module
    est_cov_diag: nn.Module
    est_cov_upper: nn.Module

    @classmethod
    def build(
        cls,
        num_latents: int,
        num_components: int = 5,
        embed_dims: list[int] = [128, 128],
        est_pi_hidden_dims: list[int] = [128, 128],
        est_mu_hidden_dims: list[int] = [128, 128],
        est_cov_diag_hidden_dims: list[int] = [128, 128],
    ):
        k, n = num_components, num_latents
        embed = MLP(embed_dims, nn.gelu)
        est_pi = MLP(est_pi_hidden_dims + [k], nn.gelu)
        est_mu = MLP(est_mu_hidden_dims + [k], nn.gelu)
        est_cov_diag = MLP(est_cov_diag_hidden_dims + [k * n])
        est_cov_upper = MLP(est_cov_diag_hidden_dims + [k * (n * (n - 1)) // 2])
        return MDN(embed, est_pi, est_mu, est_cov_diag, est_cov_upper)

    @nn.compact
    def __call__(self, x: jax.Array, **kwargs):
        h = self.embed(x)
        pi_logits = self.est_pi(h)
        mu = self.est_mu(h)
        cov_diag = self.est_cov_diag(h)
        cov_upper = self.est_cov_upper(h)

        pass
