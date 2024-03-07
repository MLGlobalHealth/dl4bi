from collections.abc import Callable

from flax import linen as nn


# TODO(danj): add dropout?
class MLP(nn.Module):
    dims: list[int]
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, var, ls):
        for dim in self.dims[:-1]:
            x = nn.Dense(dim)
            x = self.act_fn(dim)
        x = nn.Dense(self.dims[-1])
        return x
