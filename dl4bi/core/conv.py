from collections.abc import Callable
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import vmap
from jax.tree_util import Partial

from .metrics import l2_dist_sq
from .utils import pad_concat


class ConvDeepSet(nn.Module):
    """A ConvDeepSet from [The Convolutional Conditional Neural Process](https://arxiv.org/abs/1910.13556).

    The general idea is to use an RBF network to evaluate the function at
    test locations using the context locations. The density channel is used to
    normalize the convolutional output. This implementation is based on the original
    [here](https://github.com/cambridge-mlg/convcnp/blob/master/convcnp/architectures.py).

    Args:
        d_out: Dimension of output, i.e. number of output channels.
        use_density: Use a density channel for normalization.
        dtype: Data type to use for calculations.

    Returns:
        An instance of `ConvDeepSet`.
    """

    d_out: int = 8
    use_density: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,
        f_ctx: jax.Array,
        s_test: jax.Array,
        valid_lens_ctx: Optional[jax.Array] = None,
        valid_lens_test: Optional[jax.Array] = None,
        **kwargs,
    ):
        B, L_ctx, d_f = f_ctx.shape
        d_in = d_f + self.use_density
        d_sq = vmap(l2_dist_sq)(s_test, s_ctx)[..., None]  # [B, L_test, L_ctx, 1]
        log_ls = self.param("log_lengthscale", nn.initializers.constant(-1), (d_in,))
        ls = jnp.exp(log_ls)[None, None, None, :]  # [1, 1, 1, d_in]
        rbf_w = jnp.exp(-d_sq / (2 * ls**2))  # [B, L_test, L_ctx, d_in]
        f_test = f_ctx
        if self.use_density:
            density = jnp.ones((B, L_ctx, 1))
            f_test = jnp.concatenate([density, f_ctx], axis=-1)  # [B, L_ctx, d_in]
        f_test = f_test[:, None, ...] * rbf_w  # [B, L_test, L_ctx, d_in]
        if valid_lens_ctx is None:
            valid_lens_ctx = jnp.repeat(L_ctx, B)
        mask = (jnp.arange(L_ctx) < valid_lens_ctx[:, None])[:, None, :, None]
        f_test = f_test.sum(axis=2, where=mask)  # [B, L_test, d_in]
        if self.use_density:
            density, conv = f_test[..., :1], f_test[..., 1:]
            normed_conv = conv / (density + 1e-8)
            f_test = jnp.concatenate([density, normed_conv], axis=-1)
        return nn.Dense(self.d_out, dtype=self.dtype)(f_test)  # [B, L_test, d_out]


class SimpleConv(nn.Module):
    """A 4-layer convoultional network with fixed stride and channels.

    This implementation is based on the original [here](https://github.com/cambridge-mlg/convcnp/blob/master/convcnp/architectures.py).

    Args:
        dtype: Data type to use for calculations.

    Returns:
        An instance of `SimpleConv`.
    """

    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array):
        d_x = x.shape[-1]
        Conv = Partial(nn.Conv, kernel_size=5, strides=1, dtype=self.dtype)
        for n in [16, 32, 16, d_x]:
            x = nn.relu(Conv(n)(x))
        return x


class UNet(nn.Module):
    """A 12-layer residual network with skip connections using concatenation.

    This implementation is based on the original [here](https://github.com/cambridge-mlg/convcnp/blob/master/convcnp/architectures.py).

    Args:
        num_halving_layers: Number of layers to halve.
        dtype: Data type to use for calculations.

    Returns:
        An instance of `Unet`.
    """

    num_halving_layers: int = 6
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        d_x = x.shape[-1]
        Conv = Partial(
            nn.Conv,
            kernel_size=5,
            strides=2,
            padding=2,
            dtype=self.dtype,
        )
        ConvT = Partial(
            nn.ConvTranspose,
            kernel_size=5,
            strides=2,
            padding=2,
            dtype=self.dtype,
        )
        h, hs = x, [x]
        for n in [1, 2, 2, 4, 4, 8]:
            h = nn.relu(Conv(n * d_x)(h))
            hs += [h]
        h = hs.pop()
        for n in [4, 4, 2, 2, 1, 1]:
            h = nn.relu(ConvT(n * d_x)(h))
            h = pad_concat(hs.pop(), h)
        return h


class ResNetBlock(nn.Module):
    """A ResNetBlock based on Flax [example](https://github.com/google/flax/blob/main/examples/imagenet/models.py).

    Args:
        num_features: Number of features for convolutions.
        kernel: Tuple of kernel dimensions.
        strides: Tuple of stride dimenions.
        act_fn: Activation function to use.
        dtype: Data type to use for calculations.

    Returns:
        An instance of `ResNetBlock`.
    """

    num_features: int
    kernel: tuple = (3, 3)
    strides: tuple = (1, 1)
    act_fn: Callable = nn.relu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jax.Array,  # [B, ...spatial dims..., C]
        training: bool = False,
    ):
        r = x  # residual
        d = len(self.strides)  # num spatial dims
        Bn = Partial(
            nn.BatchNorm,
            use_running_average=not training,
            axis_name="batch",
            dtype=self.dtype,
        )
        Conv = Partial(
            nn.Conv,
            features=self.num_features,
            strides=self.strides,
            use_bias=False,
            dtype=self.dtype,
        )
        x = Conv(self.kernel)(x)
        x = Bn()(x)
        x = self.act_fn(x)
        x = Conv(self.kernel)(x)
        x = Bn(scale_init=nn.initializers.zeros_init())(x)
        if r.shape != x.shape:
            r = Conv(kernel_size=(1,) * d, name="conv_proj")(r)
            r = Bn(name="norm_proj")(r)
        return self.act_fn(r + x)


class ConvCNPBlock(nn.Module):
    """A depthwise-separable pre-activation ResNetBlock based on Yann Dubois' implementation [here](https://github.com/YannDubs/Neural-Process-Family/blob/master/npf/architectures/cnn.py).

    Args:
        num_features: Number of features for convolutions.
        kernel: Tuple of kernel dimensions.
        act_fn: Activation function to use.
        dtype: Data type to use for calculations.

    Returns:
        An instance of `ConvCNPBlock`.
    """

    num_features: int
    kernel: tuple = (3, 3)
    act_fn: Callable = nn.relu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jax.Array,  # [B, ...spatial dims..., C]
        training: bool = False,
    ):
        r = x
        n = x.shape[-1]
        d = len(self.kernel)  # num spatial dims
        Bn = Partial(
            nn.BatchNorm,
            use_running_average=not training,
            axis_name="batch",
            dtype=self.dtype,
        )
        DepthConv = Partial(
            nn.Conv,
            features=n,
            feature_group_count=n,
            kernel_size=self.kernel,
            use_bias=True,
            dtype=self.dtype,
        )
        PointConv = Partial(
            nn.Conv,
            kernel_size=(1,) * d,
            use_bias=True,
            dtype=self.dtype,
        )
        x = Bn()(x)
        x = self.act_fn(x)
        x = DepthConv()(x)
        x = PointConv(n)(x)
        x = Bn()(x)
        x = self.act_fn(x)
        x = DepthConv()(x)
        return PointConv(self.num_features)(x + r)


class ConvCNPNet(nn.Module):
    """A CNN using ConvCNP blocks based on on Yann Dubois' implementation [here](https://github.com/YannDubs/Neural-Process-Family/blob/master/npf/architectures/cnn.py).

    Args:
        r_out: Dimension of latent dim, r.
        kernel: A tuple of kernel dimensions.
        num_blks: Number of `ConvCNPBlock`s to use.
        dtype: Data type to use for calculations.

    Returns:
        An instance of `ConvDeepSet`.
    """

    r_dim: int = 128
    kernel: tuple = (3, 3)
    num_blks: int = 5
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jax.Array,  # [B, ...spatial dims..., C]
        training: bool = False,
    ):
        for _ in range(self.num_blks):
            x = ConvCNPBlock(self.r_dim, self.kernel, dtype=self.dtype)(x, training)
        return x


class ResNeXtBlock(nn.Module):
    """ResNeXtBlock based on [d2l](https://d2l.ai/chapter_convolutional-modern/ resnet.html)'s implementation and the Flax [example](https://github.com/google/ flax/blob/main/examples/imagenet/models.py).

    Args:
        num_features: Number of features for convolutions.
        num_groups: Number of groups to convolute.
        kernel: Tuple of kernel dimensions.
        strides: Tuple of stride dimenions.
        act_fn: Activation function to use.
        dtype: Data type to use for calculations.

    Returns:
        An instance of `ResNeXtBlock`.
    """

    num_features: int
    num_groups: int = 16
    kernel: tuple = (3, 3)
    strides: tuple = (1, 1)
    act_fn: Callable = nn.relu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jax.Array,  # [B, ...spatial dims..., C]
        training: bool = False,
    ):
        r = x  # residual
        d = len(self.kernel)  # num spatial dims
        Bn = Partial(
            nn.BatchNorm,
            use_running_average=not training,
            axis_name="batch",
            dtype=self.dtype,
        )
        Conv = Partial(nn.Conv, use_bias=False, dtype=self.dtype)
        x = Conv(self.num_features, (1,) * d)(x)
        x = Bn()(x)
        x = self.act_fn(x)
        x = Conv(
            self.num_features,
            self.kernel,
            self.strides,
            feature_group_count=self.num_features // self.num_groups,
        )(x)
        x = Bn()(x)
        x = self.act_fn(x)
        x = Conv(self.num_features * 4, (1,) * d)(x)
        x = Bn(scale_init=nn.initializers.zeros_init())(x)
        if r.shape != x.shape:
            r = Conv(
                self.num_features * 4,
                (1,) * d,
                self.strides,
                name="conv_proj",
            )(r)
            r = Bn(name="norm_proj")(r)
        return self.act_fn(r + x)


class ConvBlock(nn.Module):
    """A ConvBlock based on [d2l's implementation](https://d2l.ai/chapter_convolutional-modern/densenet.html).

    Args:
        num_features: Number of features for convolutions.
        kernel: Tuple of kernel dimensions.
        act_fn: Activation function to use.
        dtype: Data type to use for calculations.

    Returns:
        An instance of `ConvBlock`.
    """

    num_features: int
    kernel: tuple = (3, 3)
    act_fn: Callable = nn.relu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jax.Array,  # [B, ...spatial dims..., C]
        training: bool = False,
    ):
        y = nn.BatchNorm(not training, dtype=self.dtype)(x)
        y = self.act_fn(y)
        y = nn.Conv(self.num_features, self.kernel, dtype=self.dtype)(y)
        return jnp.concatenate([x, y], -1)


class DenseBlock(nn.Module):
    """A DenseBlock based on [d2l's implementation](https://d2l.ai/chapter_convolutional-modern/densenet.html).

    Args:
        num_blks: Number of convolution blocks to use.
        num_features: Number of features for convolutions.
        kernel: Tuple of kernel dimensions.
        act_fn: Activation function to use.
        dtype: Data type to use for calculations.

    Returns:
        An instance of `DenseBlock`.
    """

    num_blks: int
    num_features: int
    kernel: tuple = (3, 3)
    act_fn: Callable = nn.relu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jax.Array,  # [B, ...spatial dims..., C]
        training: bool = False,
    ):
        blk = ConvBlock(self.num_features, self.kernel, self.act_fn, self.dtype)
        for _ in range(self.num_blks):
            x = blk.copy()(x, training)
        return x


class TransitionBlock(nn.Module):
    """A TransitionBlock based on [d2l's implementation](https://d2l.ai/chapter_convolutional-modern/densenet.html).

    Args:
        num_features: Number of features for convolutions.
        act_fn: Activation function to use.
        dtype: Data type to use for calculations.

    Returns:
        An instance of `DenseBlock`.
    """

    num_features: int
    act_fn: Callable = nn.relu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jax.Array,  # [B, ...spatial dims..., C]
        training: bool = False,
    ):
        d = x.ndim - 2  # num spatial dims
        x = nn.BatchNorm(not training, dtype=self.dtype)(x)
        x = self.act_fn(x)
        x = nn.Conv(self.num_features, (1,) * d, dtype=self.dtype)(x)
        return nn.avg_pool(x, (2,) * d, (2,) * d)
