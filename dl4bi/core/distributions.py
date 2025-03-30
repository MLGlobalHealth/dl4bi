import flax.linen as nn
import jax
import jax.numpy as jnp


class DiscretizedContinuous(nn.Module):
    """
    Based on https://github.com/automl/TransformersCanDoBayesianInference/blob/master/bar_distribution.py
    """

    # TODO @pgrynfelder: unbounded case with half-normal tails

    boundaries: jax.Array
    # buckets are [a, b) where a,b = boundaries[i:i+1]
    # and the last bucket is closed
    N: int  # num buckets
    L: jax.Array
    U: jax.Array

    def setup(self, boundaries: jax.Array):
        assert boundaries.ndim == 1

        boundaries = jnp.sort(boundaries)
        self.boundaries = boundaries
        self.bucket_widths = boundaries[1:] - boundaries[:-1]
        self.N = boundaries.shape[0] - 1
        self.L = boundaries[0]
        self.U = boundaries[self.N]

    def find_bucket(self, y: jax.Array):
        bucket_ids = (
            jnp.searchsorted(self.boundaries, y, side="right", method="scan_unrolled")
            - 1
        )
        # account for the last bucket being closed
        bucket_ids = bucket_ids.at[y == self.U].set(self.N - 1)

        return bucket_ids

    def logpdf(self, logits: jax.Array, y: jax.Array):
        assert y.ndim == 1, f"y must be one-dimensional, got {y.shape}"
        bucket_ids = self.find_bucket(y)
        assert (bucket_ids >= 0).all() and (bucket_ids < self.N), (
            f"{y=} not in the support"
        )
        assert logits.shape[-1] == self.N, (
            f"last dimension of {logits.shape=} must match the number of buckets {self.N}"
        )

        bucket_logprobs = jax.nn.log_softmax(logits, axis=-1)
        logprobs = bucket_logprobs - jnp.log(self.bucket_widths)

        return logprobs[bucket_ids]
