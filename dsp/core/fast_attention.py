class MultiheadFastAttention(nn.Module):
    r"""Multihead implementation of [FAVOR+](https://arxiv.org/abs/2009.14794).

    Args:
        proj_qs: A module for projecting queries.
        proj_ks: A module for projecting keys.
        proj_vs: A module for projecting values.
        proj_out: A module for projecting output.
        build_phi: A function for buliding attention kernel.
        num_heads: Number of heads for attention module.
        num_ortho_feautres: Number of orthogonal features to use for
            fast Performer attention.
        p_dropout: A dropout rate for attention.

    Returns:
        A `MultiheadFastSoftmaxAttention` module.
    """

    proj_qs: nn.Module = MLP([64])
    proj_ks: nn.Module = MLP([64])
    proj_vs: nn.Module = MLP([64])
    proj_out: nn.Module = MLP([64])
    build_phi: Callable = build_stable_positive_softmax_phi
    num_heads: int = 4
    num_ortho_features: int = 64
    p_dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, D_QK]
        ks: jax.Array,  # [B, K, D_QK]
        vs: jax.Array,  # [B, K, D_V]
        valid_lens: Optional[jax.Array] = None,  # [B] or [B, Q]
        training: bool = False,
        rng_redraw_random_features: Optional[jax.Array] = None,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of dimension $\mathbb{R}^{B\times Q\times D_{Q,K}}$
            ks: Keys of dimension $\mathbb{R}^{B\times K\times D_{Q,K}}$
            vs: Values of dimension $\mathbb{R}^{B\times K\tiems D_V}$
            valid_lens: Mask consisting of valid length per sequence of dimension
                $\mathbb{R}^B$ or $\mathbb{R}^{B\times K}$
            training: Boolean indicating whether currently training.
            redraw_random_features: Redraw random features used for kernel
                approximation of attention.

        Returns:
            `ctx` and `attn`, the updated values and None, respectively,
            since the attention matrix is never materialized in FAVOR+.
        """
        qs, ks, vs = self.proj_qs(qs), self.proj_ks(ks), self.proj_vs(vs)
        (B, Q, D_QK), K, D_V, H = qs.shape, ks.shape[1], vs.shape[-1], self.num_heads
        D_QK_H, D_V_H = D_QK // H, D_V // H
        # [B, {Q,K}, D_{QK,V}] -> [B * H, {Q,K}, D_{QK,V}_H]
        qs = qs.reshape(B, Q, H, D_QK_H).transpose(0, 2, 1, 3).reshape(-1, Q, D_QK_H)
        ks = ks.reshape(B, K, H, D_QK_H).transpose(0, 2, 1, 3).reshape(-1, K, D_QK_H)
        vs = vs.reshape(B, K, H, D_V_H).transpose(0, 2, 1, 3).reshape(-1, K, D_V_H)
        if valid_lens is not None:
            valid_lens = jnp.repeat(valid_lens, H, axis=0)
        ctx, attn = FastAttention(
            self.p_dropout, self.build_phi, self.num_ortho_features
        )(qs, ks, vs, valid_lens, training, rng_redraw_random_features)
        ctx = ctx.reshape(B, H, Q, D_V_H).transpose(0, 2, 1, 3).reshape(B, Q, D_V)
        return self.proj_out(ctx), attn
