import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from sps.kernels import rbf
from sps.utils import build_grid

from dl4bi.core import RBFRandomFourierFeatures


def test_rbf_random_fourier_features():
    rng = random.key(55)
    B, L, H, E = 3, 103, 4, 48
    s = build_grid([{"start": -2.5, "stop": 2.5, "num": L}])
    s = jnp.repeat(s[None, ...], B, axis=0)
    rbf_rff = RBFRandomFourierFeatures(E, H)
    (s_rff), _ = rbf_rff.init_with_output(rng, s)
    assert s_rff.shape == (B, L, H, E), "Incorrect dimensions for RBF RFF embedding!"
    rbf_dist = rbf(s[0], s[0], var=1.0, ls=1.0)
    s_rff_0 = s_rff[0, :, 0, :]
    rff_dist = jnp.einsum("A D, B D -> A B", s_rff_0, s_rff_0)
    abs_error = jnp.abs(rbf_dist - rff_dist)
    assert jnp.max(abs_error) < 0.3, "Large max error for RBF distance!"
    # # plot actual dist
    # plt.imshow(rff_dist, cmap="Spectral_r")
    # plt.title("RFF Distance")
    # plt.colorbar()
    # plt.savefig("/tmp/rff_dist.png")
    # plt.clf()
    # # plot errors
    # plt.imshow(abs_error, cmap="Spectral_r")
    # plt.title("Absolute Error of RBF - RFF Distance")
    # plt.colorbar()
    # plt.savefig("/tmp/rbf_rff_dist.png")
    # plt.clf()
