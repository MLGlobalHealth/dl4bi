from dataclasses import dataclass

import jax

from ..core.data import Batch


@dataclass(frozen=True)
class RegressionBatch(Batch):
    x: jax.Array
    y: jax.Array


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    RegressionBatch,
    lambda d: ((d.x, d.y), None),
    lambda _aux, children: RegressionBatch(*children),
)
