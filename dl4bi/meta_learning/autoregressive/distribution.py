import jax
import jax.numpy as jnp
from numpyro.distributions import Distribution

from .sampler import ARSampler


class ARDistribution(Distribution):
    """
    Numpyro distribution for
    `f_test | f_test, s_ctx, f_ctx`.
    """

    sampler: ARSampler
    s_ctx: jax.Array
    f_ctx: jax.Array
    s_test: jax.Array

    def __init__(
        self,
        sampler: ARSampler,
        s_ctx: jax.Array,  # [L_ctx, Ds]
        f_ctx: jax.Array,  # [L_ctx, Df]
        s_test: jax.Array,  # [L_test, Ds]
        batch_size: int,
    ):
        s_ctx, f_ctx = jnp.array(s_ctx), jnp.array(f_ctx)
        _, Df = f_ctx.shape
        L_test, _ = s_test.shape

        self.sampler = sampler
        self.s_ctx = s_ctx[None].repeat(batch_size)
        self.f_ctx = f_ctx[None].repeat(batch_size)
        self.s_test = s_test[None].repeat(batch_size)

        super().__init__(batch_shape=(batch_size,), event_shape=(L_test, Df))

    def sample(self, rng, sample_shape=()):
        if sample_shape != ():
            raise NotImplementedError

        sample, logprobs = self.sampler.sample(
            rng, self.s_ctx, self.f_ctx, self.s_test, strategy=None
        )
        return sample

    def log_prob(self, value):
        rng = jax.random.seed(0)
        return self.sampler.logpdf(
            rng,  # unused for deterministic sampling strategies
            self.s_ctx,
            self.f_ctx,
            self.s_test,
            value,
            strategy=None,
        )


class PresampledDistribution(Distribution):
    """
    Distribution operating on pre-sampled data.

    The idea is that if we have a distribution thats independent from the rest of the model,
    then presampling N samples ahead of time `samples = [s_0, ..., s_N-1]`
    and outputting them in order is the same as sampling within the program.

    To associate logprobs with the samples they are identified with an integer.
    `sample` will output integers.
    To get the actual sample value, call `get_value`
    """

    samples: jax.Array
    logprobs: jax.Array

    def __init__(
        self,
        S,  # [N, ...]
        logprobs,  # [N]
        batch_size: int,
    ):
        self.samples = S
        self.logprobs = logprobs
        self.batch_size = batch_size

        super().__init__(batch_shape=(batch_size,), event_shape=())

    def sample(self, rng, sample_shape=()):
        if sample_shape != ():
            raise NotImplementedError

        idx = jnp.arange(self.i, self.i + self.batch_size)
        self.i += self.batch_size
        return idx

    def log_prob(self, idx):
        return self.logprobs[idx]

    def get_value(self, idx):
        return self.samples[idx]
