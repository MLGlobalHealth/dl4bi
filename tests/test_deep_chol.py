from dge import DeepChol
from sps.gp import cholesky
from sps.kernels import rbf
from sps.utils import build_grid


def test_deep_chol():
    locs = build_grid([{"start": 0, "stop": 1, "num": 32}])
