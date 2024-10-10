import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from sps.gp import GP
from sps.kernels import matern_3_2, periodic, rbf
from sps.priors import Prior

import dl4bi.meta_regression.train_utils as tu


def kernel_fn_to_plot_name(kernel_fn):
    plot_name = {rbf: "RBF", matern_3_2: "Matern 3/2", periodic: "Periodic"}.get(
        kernel_fn, None
    )
    if plot_name is None:
        raise ValueError(f"Unidentified kernel function used: {kernel_fn.__name__}")
    return plot_name


def kernel_name_to_fn(kernel_names: list[str]):
    kernel_fns = []
    valid_kernels = {k.__name__: k for k in [rbf, matern_3_2, periodic]}
    for kernel_name in kernel_names:
        if kernel_name not in valid_kernels:
            raise ValueError(f"Unidentified kernel name used: {kernel_name}")
        kernel_fns.append(valid_kernels[kernel_name])
    return kernel_fns


def get_models(ckpt_base_dir: str, model_seed: int, model_name: str, kernels: list):
    return {
        kernel_fn.__name__: tu.load_ckpt(
            Path(ckpt_base_dir) / f"{kernel_fn.__name__}/{model_seed}/{model_name}.ckpt"
        )[0]
        for kernel_fn in kernels
    }


def sample(
    num_ctx,
    rng_sample,
    kernel_fn,
    ls,
    rng_gp,
    rng_noise,
    obs_noise=0.1,
    period=1.5,
):
    s_min, s_max, num_test = -2, 2, 128
    s_test = jnp.linspace(s_min, s_max, num_test)[:, None]
    s_ctx = random.uniform(rng_sample, minval=s_min, maxval=s_max, shape=(num_ctx, 1))
    if kernel_fn == periodic:
        gp = GP(
            kernel_fn,
            ls=Prior("fixed", {"value": ls}),
            period=Prior("fixed", {"value": period}),
        )
    else:
        gp = GP(kernel_fn, ls=Prior("fixed", {"value": ls}))
    f, *_ = gp.simulate(rng_gp, jnp.vstack([s_ctx, s_test]), batch_size=1)
    f_ctx, f_test = f[0, :num_ctx, :], f[0, num_ctx:, :]
    f_ctx = f_ctx + obs_noise * random.normal(rng_noise, f_ctx.shape)
    return s_ctx, f_ctx, s_test, f_test


def plot_kernel_lengthscale_posterior(
    ckpt_base_dir: str,
    output_dir_base: str,
    model_seed: int = 20,
    model_name: str = "TNPKR Fast",
    kernels: list = [rbf, matern_3_2, periodic],
    lengthscales: list[float] = [0.1, 0.3, 0.5],
    num_ctx: int = 10,
    num_repeats: int = 1,
):
    output_dir = (Path(output_dir_base) / model_name) / str(num_ctx)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = get_models(ckpt_base_dir, model_seed, model_name, kernels)

    for example in range(num_repeats):
        rng = random.key(example)
        plt.clf()
        fig, axs = plt.subplots(
            len(kernels),
            len(lengthscales),
            figsize=(6 * len(lengthscales), 4 * len(kernels)),
        )
        rng_gp, rng_extra, rng_sample, rng_noise = random.split(rng, 4)
        for i, kernel_fn in enumerate(kernels):
            state = models[kernel_fn.__name__]
            for j, ls in enumerate(lengthscales):
                s_ctx, f_ctx, s_test, f_test = sample(
                    num_ctx,
                    rng_sample,
                    kernel_fn,
                    ls,
                    rng_gp,
                    rng_noise,
                )
                f_mu, f_std, *_ = state.apply_fn(
                    {"params": state.params, **state.kwargs},
                    s_ctx[None, ...],
                    f_ctx[None, ...],
                    s_test[None, ...],
                    rngs={"extra": rng_extra},
                )

                plt.sca(axs[j, i])
                tu.plot_posterior_predictive(
                    s_ctx.squeeze(),
                    f_ctx.squeeze(),
                    s_test.squeeze(),
                    f_test.squeeze(),
                    f_mu.squeeze(),
                    f_std.squeeze(),
                )
                axs[j, i].set_title(
                    f"{kernel_fn_to_plot_name(kernel_fn)}" if j == 0 else ""
                )
                axs[j, i].set_ylabel(f"LS={ls}" if i == 0 else "")
                axs[j, i].set_xlabel("s" if j == len(lengthscales) - 1 else "")

        plot_path = output_dir / f"plot_{num_ctx}_{example}.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)


def reproduce_posterior_predictive_plots(args):
    for model_name in args.models:
        for num_ctx in args.num_ctx:
            plot_kernel_lengthscale_posterior(
                ckpt_base_dir=args.ckpt_base_dir,
                output_dir_base=args.output_base_dir,
                num_ctx=num_ctx,
                model_name=model_name,
                model_seed=args.model_seed,
                kernels=args.kernels,
                num_repeats=args.num_repeats,
            )


def get_args():
    parser = argparse.ArgumentParser(description="Posterior Predictive Plot Generation")
    parser.add_argument(
        "-c",
        "--ckpt_base_dir",
        type=str,
        required=True,
        help="Base directory containing model checkpoints"
        " The script expects the models to be saved as <base>/<kernel>/<model_seed>/<model_name>.ckpt",
    )
    parser.add_argument(
        "-o",
        "--output_base_dir",
        type=str,
        required=True,
        help="Base directory for saving generated plots."
        " The script will save the plots by <out_base>/<model_name>/<num_ctx>/plot_<num_ctx>_<num_repeat>.png",
    )
    parser.add_argument(
        "-m",
        "--models",
        type=str,
        nargs="+",
        default=["TNPKR Fast"],
        help="List of model names to plot",
    )
    parser.add_argument(
        "-n",
        "--num_ctx",
        type=int,
        nargs="+",
        default=[15],
        help="List of num_ctx values to process",
    )
    parser.add_argument(
        "-r",
        "--num_repeats",
        type=int,
        default=100,
        help="Number of times to repeat each plot generation",
    )
    parser.add_argument(
        "-s",
        "--model_seed",
        type=int,
        default=20,
        help="The seed of the model used to plot",
    )
    parser.add_argument(
        "-k",
        "--kernels",
        type=str,
        nargs="+",
        default=["rbf", "matern_3_2", "periodic"],
        help="List of kernels to use",
    )
    args = parser.parse_args()
    args.kernels = kernel_name_to_fn(args.kernels)
    return args


if __name__ == "__main__":
    args = get_args()
    reproduce_posterior_predictive_plots(args)
