from functools import partial

import jax
from jax import jit


@partial(jit, static_argnames=["batch_size"])
def batch(x: jax.Array, batch_size: int):
    N, *dims = x.shape
    N_rounded = N // batch_size * batch_size
    if N != N_rounded:
        # this is ok being printed at compile time only
        print(
            f"batch_size does not divide the number of samples. Dropping final {N - N_rounded} samples."
        )
    x = x[:N_rounded].reshape(-1, batch_size, *dims)
    return x


@jit
def unbatch(x: jax.Array):
    N, B, *dims = x.shape
    x = x.reshape(-1, *dims)
    return x
