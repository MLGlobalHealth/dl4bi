from collections.abc import Callable

from flax import linen as nn


class MLP(nn.Module):
    dims: list[int]
    act_fn: Callable = nn.selu
    p_dropout: float = 0.5

    @nn.compact
    def __call__(self, x, training: bool = False):
        for dim in self.dims[:-1]:
            x = nn.Dense(dim)(x)
            x = self.act_fn(x)
            x = nn.Dropout(self.p_dropout, deterministic=not training)(x)
        return nn.Dense(self.dims[-1])(x)
