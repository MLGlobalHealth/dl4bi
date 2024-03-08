from collections.abc import Callable

from flax import linen as nn


# TODO(danj): recode from here https://github.com/MLGlobalHealth/pi-vae/blob/main/src_py/models/pivae.py
class Phi(nn.Module):
    dims: list[int]
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        for dim in self.dims[:-1]:
            x = nn.Dense(dim)(x)
            x = self.act_fn(x)
        return nn.Dense(self.dims[-1])(x)
