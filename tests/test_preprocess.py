from jax import random

from dl4bi.core.preprocess import Whitener, condition_number


def test_whitener():
    rng = random.key(55)
    X = random.normal(rng, (1024, 16))
    Xw = Whitener().fit_transform(X)
    err_msg = "Whitening did not improve conditioning!"
    assert condition_number(Xw) < condition_number(X), err_msg
