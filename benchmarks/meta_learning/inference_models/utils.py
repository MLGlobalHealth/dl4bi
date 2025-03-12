import importlib
import math
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from jax import jit, random
from jax.scipy.stats import norm
from numpyro.infer import MCMC, NUTS
from omegaconf import DictConfig
from sps.kernels import rbf


def collect_infer_funcs(infer_model_name: str, data: DictConfig):
    module = importlib.import_module(f"inference_models.{infer_model_name}")
    prior_pred = getattr(module, "jax_prior_pred", module.numpyro_prior_pred)
    dataloader = module.build_dataloader(prior_pred, data)
    return (
        dataloader,
        module.batch_to_infer_kwargs,
        module.numpyro_model,
        module.numpyro_pointwise_post_pred,
        module.visualize,
    )


# TODO(danj): verify this formula
@partial(jit, static_argnames=("jitter", "kernel"))
def condition_gp(
    rng: jax.Array,
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    s_test: jax.Array,
    var: jax.Array,
    ls: jax.Array,
    kernel: Callable = rbf,
    jitter=1e-5,
):
    """Conditions a GP on observed points.

    Source: https://num.pyro.ai/en/stable/examples/gp.html
    """
    L_ctx, L_test = s_ctx.shape[0], s_test.shape[0]
    k_tt = kernel(s_test, s_test, var, ls) + jitter * jnp.eye(L_test)
    k_tc = kernel(s_test, s_ctx, var, ls)
    k_cc = kernel(s_ctx, s_ctx, var, ls) + jitter * jnp.eye(L_ctx)
    K_cc_cho = jax.scipy.linalg.cho_factor(k_cc)
    K = k_tt - k_tc @ jax.scipy.linalg.cho_solve(K_cc_cho, k_tc.T)
    mu = k_tc @ jax.scipy.linalg.cho_solve(K_cc_cho, f_ctx)
    noise = jnp.sqrt(jnp.clip(jnp.diag(K), 0.0)) * random.normal(rng, L_test)
    return mu + noise


def run_hmc(rng: jax.Array, model: Callable, infer: DictConfig, **kwargs):
    mcmc = MCMC(
        NUTS(model),
        num_warmup=infer.num_warmup,
        num_samples=infer.num_samples,
        num_chains=infer.num_chains,
    )
    mcmc.run(rng, **kwargs)
    return mcmc


def visualize_spatial(**kwargs):
    if kwargs["s_ctx"].ndim == 1:  # 1D
        return plot_1d_posterior_predictive(**kwargs)
    return plot_2d_posterior_predictive(**kwargs)


def plot_1d_posterior_predictive(
    s_ctx: jax.Array,  # [L_ctx]
    f_ctx: jax.Array,  # [L_ctx]
    s: jax.Array,  # [L]
    f: jax.Array,  # [L]
    f_mu_model: jax.Array,  # [L]
    f_std_model: jax.Array,  # [L]
    f_mu_hmc: jax.Array,  # [L]
    f_std_hmc: jax.Array,  # [L]
    inv_permute_idx: jax.Array,  # [L]
    hdi_prob: float = 0.95,
    **kwargs,
):
    # palette from https://davidmathlogic.com/colorblind
    magenta, green, blue, gold = "#D81B60", "#D81B60", "#1E88E5", "#FFC107"
    plt.scatter(s_ctx, f_ctx, color=magenta)
    order = lambda x: x[inv_permute_idx]
    s = order(s)
    plt.plot(s, order(f), color=magenta)
    _plot_1d_bounds(
        s,
        order(f_mu_model),
        order(f_std_model),
        color=blue,
        hdi_prob=hdi_prob,
    )
    _plot_1d_bounds(
        s,
        order(f_mu_hmc),
        order(f_std_hmc),
        color=gold,
        hdi_prob=hdi_prob,
    )
    plt.xlabel("s")
    plt.ylabel("f")
    return plt.gcf()


def _plot_1d_bounds(
    s: jax.Array,
    f_mu: jax.Array,
    f_std: jax.Array,
    color: str = "steelblue",
    hdi_prob: float = 0.95,
):
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    f_lower = f_mu - z_score * f_std
    f_upper = f_mu + z_score * f_std
    plt.plot(s, f_mu, color=color)
    plt.fill_between(s, f_lower, f_upper, alpha=0.4, color=color, interpolate=True)


def plot_2d_posterior_predictive(
    s_ctx: jax.Array,  # [L_ctx, 2]
    f_ctx: jax.Array,  # [L_ctx]
    s: jax.Array,  # [L, 2]
    f: jax.Array,  # [L]
    f_mu_model: jax.Array,  # [L]
    f_std_model: jax.Array,  # [L]
    f_mu_hmc: jax.Array,  # [L]
    f_std_hmc: jax.Array,  # [L]
    inv_permute_idx: jax.Array,  # [L]
    shape: Optional[tuple[int, int]] = None,
    fontsize: int = 20,
    **kwargs,
):
    L_ctx, L = s_ctx.shape[0], s.shape[0]
    D = int(math.sqrt(L))
    H, W = (D, D) if shape is None else shape
    cmap, cmap_std = "viridis", "plasma"
    _, axs = plt.subplots(2, 4, figsize=(20, 10))
    task = jnp.hstack([f_ctx, jnp.repeat(jnp.nan, L - L_ctx)])
    to_img = lambda x: x[inv_permute_idx].reshape(H, W)
    min_std = min(f_std_hmc.min(), f_std_model.min())
    max_std = max(f_std_hmc.max(), f_std_model.max())
    norm_std = mpl.colors.Normalize(vmin=min_std, vmax=max_std)
    task, f = to_img(task), to_img(f)
    axs[0, 0].set_ylabel("HMC", fontsize=fontsize)
    axs[0, 0].set_title("Task", fontsize=fontsize)
    axs[0, 0].imshow(task, cmap=cmap, interpolation="none")
    axs[0, 1].set_title("Uncertainty", fontsize=fontsize)
    axs[0, 1].imshow(
        to_img(f_std_hmc), cmap=cmap_std, norm=norm_std, interpolation="none"
    )
    axs[0, 2].set_title("Prediction", fontsize=fontsize)
    axs[0, 2].imshow(to_img(f_mu_hmc), cmap=cmap, interpolation="none")
    axs[0, 3].set_title("Ground Truth", fontsize=fontsize)
    axs[0, 3].imshow(f, cmap=cmap, interpolation="none")
    axs[1, 0].set_ylabel("TNP-KR", fontsize=fontsize)
    axs[1, 0].imshow(task, cmap=cmap, interpolation="none")
    axs[1, 1].imshow(
        to_img(f_std_model), cmap=cmap_std, norm=norm_std, interpolation="none"
    )
    axs[1, 2].imshow(to_img(f_mu_model), cmap=cmap, interpolation="none")
    axs[1, 3].imshow(f)
    return plt.gcf()
