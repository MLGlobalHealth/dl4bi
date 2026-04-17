import argparse
import sys

sys.path.append("benchmarks/vae")

from pathlib import Path
from typing import Callable, Optional, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import optax
from jax import Array, jit, random
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from dl4bi_sps.kernels import matern_1_2
from dl4bi_sps.utils import build_grid
from utils.plot_utils import plot_infer_trace

import wandb
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, train
from dl4bi.vae import gMLPDeepRV
from dl4bi.vae.train_utils import (
    deep_rv_train_step,
    generate_surrogate_decoder,
)


def main(seed=57, gt_ls=20):
    run_example(
        seed=seed,
        gt_ls=gt_ls,
        nn_model=gMLPDeepRV(num_blks=2),
        train_step=deep_rv_train_step,
        valid_step_fn=valid_step,
        model_label="DeepRV",
        decode_fn=lambda x: x,
    )


def run_example(
    seed: int,
    gt_ls: float,
    nn_model,
    train_step: Callable,
    valid_step_fn: Callable,
    model_label: str,
    decode_fn: Callable,
):
    # NOTE: generate seeds and directories.
    rng = random.key(seed)
    rng_train, rng_infer, rng_idxs, rng_obs, rng = random.split(rng, 5)
    wandb.init(mode="disabled")
    save_dir = Path(f"results/DeepRV_example/{model_label.replace(' ', '_')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: generates the spatial grid to train and infer on
    s = build_grid([{"start": 0.0, "stop": 100.0, "num": 16}] * 2).reshape(-1, 2)
    # NOTE: Priors for training and inference
    priors = {"ls": dist.Uniform(1.0, 100.0), "beta": dist.Normal()}
    sqrt_N = int(jnp.sqrt(s.shape[0]))
    # NOTE: The observed outcome to perform inference on
    y_obs = gen_y_obs(rng_obs, s, gt_ls)
    # NOTE: Mask detailing which locations are observable
    obs_mask = gen_spatial_obs_mask(rng_idxs, (sqrt_N, sqrt_N), obs_ratio=0.7)
    infer_model = inference_model(s, priors, decode_fn)
    optimizer = optax.adamw(cosine_annealing_lr(100_000, 1e-3), weight_decay=1e-2)
    optimizer = optax.chain(optax.clip_by_global_norm(3.0), optimizer)
    loader = gen_train_dataloader(s, priors)
    state = train(
        rng_train,
        nn_model,
        optimizer,
        train_step,
        100_000,
        loader,
        valid_step_fn,
        25_000,
        5_000,
        loader,
        return_state="best",
        valid_monitor_metric="norm MSE",
    )
    surrogate_decoder = generate_surrogate_decoder(state, nn_model)
    # NOTE: Inference DeepRV
    samples_drv, mcmc_drv, y_hat_drv = hmc(
        rng_infer, infer_model, y_obs, obs_mask, surrogate_decoder
    )
    cond_names = list(priors.keys())
    # NOTE: Plotting inference traces, and mean predictions
    plot_infer_trace(
        samples_drv, mcmc_drv, None, cond_names, save_dir / "infer_trace_drv.png"
    )
    plot_models_predictive_means(
        sqrt_N, y_obs, [y_hat_drv], obs_mask, [model_label], save_dir / "obs_means.png"
    )


def hmc(
    rng: Array,
    model: Callable,
    y_obs: Array,
    obs_mask: Union[bool, Array],
    surrogate_decoder: Optional[Callable] = None,
):
    """runs HMC on given inference model and observed f"""
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
    mcmc = MCMC(nuts, num_chains=2, num_samples=1_000, num_warmup=1_000)
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask, y=y_obs)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    post = Predictive(model, samples)(k2, surrogate_decoder=surrogate_decoder)
    return samples, mcmc, post["obs"]


def gen_train_dataloader(s: Array, priors: dict, batch_size=32):
    jitter = 5e-4 * jnp.eye(s.shape[0])
    kernel_jit = jit(lambda s, var, ls: matern_1_2(s, s, var, ls) + jitter)
    f_jit = jit(lambda L, z: jnp.einsum("ij,bj->bi", L, z))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z = random.split(rng_data, 3)
            var = 1.0
            ls = priors["ls"].sample(rng_ls)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s.shape[0]))
            K = kernel_jit(s, var, ls)
            L = jnp.linalg.cholesky(K)
            yield {"s": s, "z": z, "conditionals": jnp.array([ls]), "f": f_jit(L, z)}

    return dataloader


def inference_model(s: Array, priors: dict, decode_fn: Callable):
    """
    Builds a poisson likelihood inference model for GP and surrogate models
    """
    surrogate_kwargs = {"s": s}

    def poisson(surrogate_decoder=None, obs_mask=True, y=None):
        var = 1.0
        ls = numpyro.sample("ls", priors["ls"], sample_shape=())
        beta = numpyro.sample("beta", priors["beta"], sample_shape=())
        z = numpyro.sample("z", dist.Normal(), sample_shape=(1, s.shape[0]))
        if surrogate_decoder is None:
            K = matern_1_2(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
            L_chol = jnp.linalg.cholesky(K)
            mu = numpyro.deterministic("mu", jnp.matmul(L_chol, z[0]))
        else:  # NOTE: whether to use a replacment for the GP
            mu = numpyro.deterministic(
                "mu",
                decode_fn(
                    surrogate_decoder(z, jnp.array([ls]), **surrogate_kwargs)
                ).squeeze(),
            )
        lambda_ = jnp.exp(beta + mu)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Poisson(rate=lambda_), obs=y)

    return poisson


@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}


def gen_y_obs(rng: Array, s: Array, gt_ls: float):
    """generates a poisson observed data sample for inference"""
    rng_mu, rng_poiss = random.split(rng)
    var, ls, beta = 1.0, gt_ls, 1.0
    K = matern_1_2(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
    mu = dist.MultivariateNormal(0.0, K).sample(rng_mu)
    lambda_ = jnp.exp(beta + mu)
    return dist.Poisson(rate=lambda_).sample(rng_poiss)


def gen_spatial_obs_mask(rng: Array, grid_shape: tuple, obs_ratio: float = 0.15):
    """
    Generates a spatial observation mask for a 2D grid. Keeps a certain percentage of the domain unmasked,
    in the form of a few spatially-contiguous elliptical blobs. The output is a 1D boolean mask indicating
    which locations are observed.

    Args:
        rng: JAX PRNG key
        y_obs: Flattened signal (N,)
        grid_shape: Tuple (H, W) for reshaping the 1D signal
        obs_ratio: Fraction of the total grid to remain observed

    Returns:
        mask_flat: Flattened boolean mask of shape (N,), where True = observed, False = masked
    """
    H, W = grid_shape
    total_points = H * W
    num_obs_points = int(obs_ratio * total_points)
    mask = jnp.zeros((H, W), dtype=bool)

    points_collected = 0
    blob_idx = 0
    while points_collected < num_obs_points:
        rng_blob, rng = random.split(rng)
        rngs = random.split(rng_blob, 4)
        center_x = random.randint(rngs[0], (), 0, H)
        center_y = random.randint(rngs[1], (), 0, W)
        radius_x = random.randint(rngs[2], (), H // 8, H // 4)
        radius_y = random.randint(rngs[3], (), W // 8, W // 4)
        yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
        ellipse = (
            ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2
        ) <= 1.0
        new_mask = jnp.logical_or(mask, ellipse)
        added = jnp.sum(new_mask) - jnp.sum(mask)
        mask = new_mask
        points_collected += int(added)
        blob_idx += 1
    # NOTE: If we overshot, randomly drop extras
    if points_collected > num_obs_points:
        flat_idxs = jnp.argwhere(mask.flatten()).squeeze()
        rng_trim, _ = random.split(rngs[-1])
        selected = random.choice(
            rng_trim, flat_idxs, shape=(num_obs_points,), replace=False
        )
        final_mask = jnp.zeros(total_points, dtype=bool).at[selected].set(True)
    else:
        final_mask = mask.flatten()

    return final_mask


def plot_models_predictive_means(
    grid_size, f_obs, f_hats, obs_mask, model_names, save_path: Path, log=True
):
    f_hat_means = [
        f_mean.mean(axis=0).reshape(grid_size, grid_size) for f_mean in f_hats
    ]
    f_obs = f_obs.reshape(grid_size, grid_size)
    if log:
        f_hat_means = [jnp.log(f + 1) for f in f_hat_means]
        f_obs = jnp.log(f_obs + 1)
    vmin = jnp.min(jnp.array([f_mean.min() for f_mean in f_hat_means])).item()
    vmax = jnp.max(jnp.array([f_mean.max() for f_mean in f_hat_means])).item()
    cols = 3
    rows = int(jnp.ceil((len(f_hat_means) + 2) / cols))
    fig, ax = plt.subplots(
        rows, cols, figsize=(6 * cols, 7 * rows), constrained_layout=True
    )
    ax = ax.flatten()
    masked_f_obs = np.ma.masked_where(~obs_mask.reshape(grid_size, grid_size), f_obs)
    cmap = plt.cm.viridis
    cmap.set_bad(color="black")
    ax[0].imshow(masked_f_obs, origin="lower", cmap=cmap)
    ax[0].set_title("y observed")
    ax[1].imshow(f_obs, vmin=vmin, vmax=vmax, origin="lower", cmap=cmap)
    ax[1].set_title("y")
    for i, f_mean in enumerate(f_hat_means, start=2):
        model_name = model_names[i - 2]
        im = ax[i].imshow(f_mean, vmin=vmin, vmax=vmax, origin="lower", cmap=cmap)
        ax[i].set_title("Mean " r"$\hat{y}$" f" {model_name}")
    for i in range(len(ax)):
        ax[i].set_axis_off()
        if (i + 1) % cols == 0:
            fig.colorbar(im, ax=ax[i])
    fig.savefig(save_path, dpi=200)
    plt.clf()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--gt-ls", type=float, default=20.0)
    args = parser.parse_args()
    main(args.seed, args.gt_ls)
