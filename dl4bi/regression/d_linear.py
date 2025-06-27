from typing import Callable, Sequence

import jax.numpy as jnp
from flax import linen as nn

from ..core.utils import causal_moving_average
from .steps import likelihood_train_step, likelihood_valid_step


class DLinear(nn.Module):
    """
    DLinear from "Are Transformers Effective for Time Series Forecasting?"[https://github.com/vivva/DLinear].

    Args:
        num_output: Number of outputs, e.g. number of forecasted timesteps or
            number of forecasted timesteps * number of outputs per timestep.
        seasonal_lags: A sequence of integers representing various lags, e.g.
            `seasonal_lags=[7, 365]` would create weekly and yearly lags with
            daily data.
        lag_fn: A way to average to create "seasonal" trends, e.g.
            `causal_moving_average` or `edge_filled_centered_moving_average`.
        output_fn: Can be used for transforming the output, e.g. something
            in the `ModelOutput` family.
        train_step: What training step to use.
        valid_step: What validation step to use.

    Returns:
        An instance of `DLinear`.

    .. note::
        To get `NLinear` behavior, simply pass an empty sequence to `seasonal_lags`.
    """

    num_output: int
    seasonal_lags: Sequence[int]
    lag_fn: Callable = causal_moving_average
    output_fn: Callable = lambda x: x
    train_step: Callable = likelihood_train_step
    valid_step: Callable = likelihood_valid_step

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:  # x: [B, L]
        output = jnp.zeros((self.num_output,))
        for seasonal_lag in sorted(self.seasonal_lags):
            m = self.lag_fn(x, seasonal_lag)
            output += nn.Dense(self.num_output)(m)
            x = x[:, -m.shape[1] :] - m
        output += nn.Dense(self.num_output)(x)
        return self.output_fn(output)
