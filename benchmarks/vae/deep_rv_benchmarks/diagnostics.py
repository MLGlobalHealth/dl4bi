import sys

sys.path.append("benchmarks/vae")
from functools import partial
from pathlib import Path
from typing import Callable, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import Array, jit, random, vmap
from jax.scipy.linalg import solve_triangular
from numpyro import distributions as dist
from numpyro import sample
from numpyro.handlers import seed, substitute, trace

from dl4bi.core.train import PyTreeCheckpointer, TrainState, cosine_annealing_lr
from dl4bi.vae.train_utils import generate_surrogate_decoder


def compare_grads(
    models_dict: dict[str, nn.Module],
    s: Array,
    infer_model_fn: Callable,
    priors: dict[str, dist.Distribution],
    obs_mask: Union[bool, Array],
    num_points: int,
    save_dir: Path,
    target: str = "f",
    base_model: str = "Inducing Points",
    max_ls=100.0,
):
    plot_dir = save_dir / f"grad_plots_{target}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = random.key(0)
    surr_models = get_surr_decs(models_dict, save_dir)
    results_dict = {}
    for model_name in models_dict.keys():
        infer_model, _, _ = infer_model_fn(s, priors, obs_mask, num_points)
        surr_dec = surr_models.get(model_name, None)
        results_dict[model_name] = compute_gradients_for_target(
            infer_model, rng, num_points, target, surr_dec, max_ls
        )
    # Baseline to compare to
    baseline_vals, baseline_grads, _ = results_dict[base_model]
    print(f"\n=== MSE of grads and outputs for target: {target} ===\n")
    for model_name, (vals, grads_list, _) in results_dict.items():
        if model_name == base_model:
            continue
        val_mse = jnp.mean((jnp.stack(vals) - jnp.stack(baseline_vals)) ** 2)
        dz_mse = [
            (g[f"d{target}_dz"] - b[f"d{target}_dz"]) ** 2
            for g, b in zip(grads_list, baseline_grads)
        ]
        dz_mse = jnp.mean(jnp.stack(dz_mse))
        dls_mse = [
            (g[f"d{target}_dls"] - b[f"d{target}_dls"]) ** 2
            for g, b in zip(grads_list, baseline_grads)
        ]
        dls_mse = jnp.mean(jnp.stack(dls_mse))
        print(
            f"Model: {model_name}\n"
            f"Avg MSE ({target}): {val_mse:.4e}\n"
            f"Avg MSE (d{target}/dz): {dz_mse:.4e}\n"
            f"Avg MSE (d{target}/dls): {dls_mse:.4e}\n"
        )
    visualize_gradients_generalized(results_dict, plot_dir, target, base_model)


def compute_gradients_for_target(
    infer_model, rng_key, num_points, target, surrogate_decoder, max_ls
):
    f_vals = []
    grads_list = []
    key_model, key_z = jax.random.split(rng_key)
    z = sample("z", dist.Normal(), rng_key=key_z, sample_shape=(1, num_points))
    ls_grid = jnp.linspace(1.0, max_ls, 15)

    def wrapped_model(z_, ls_):
        substitutions = {"z": z_, "ls": ls_}
        with seed(rng_seed=key_model), substitute(data=substitutions), trace() as tr:
            infer_model(surrogate_decoder=surrogate_decoder)
        return tr[target]["value"]

    for ls_val in ls_grid:
        ls = jnp.array(ls_val)
        target_val = wrapped_model(z, ls)
        grad_dz = jax.grad(lambda z_: jnp.sum(wrapped_model(z_, ls)))(z)
        grad_dls = jax.grad(lambda ls_: jnp.sum(wrapped_model(z, ls_)))(ls)
        f_vals.append(target_val)
        grads_list.append(
            {
                f"d{target}_dz": grad_dz,
                f"d{target}_dls": grad_dls,
            }
        )

    return f_vals, grads_list, ls_grid


def visualize_gradients_generalized(
    results_dict, plot_dir: Path, target: str, base_model: str = "Inducing Points"
):
    baseline_vals, _, ls_grid = results_dict[base_model]
    # 1. MSE vs baseline for target
    plt.figure(figsize=(8, 6))
    for model_name, (vals, _, _) in results_dict.items():
        if model_name == base_model:
            continue
        mse_list = []
        for v, v_base in zip(vals, baseline_vals):
            mse = jnp.mean((v - v_base) ** 2)
            mse_list.append(mse)
        plt.plot(ls_grid, mse_list, label=model_name, marker="o")
    plt.title(f"Avg MSE ({target} vs Baseline) over ls")
    plt.xlabel("ls")
    plt.ylabel(f"MSE on {target}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"mse_vs_ls_{target}.png")
    plt.close()

    # 2. ∥d(target)/dz∥ norm
    plt.figure(figsize=(8, 6))
    for model_name, (_, grads_list, _) in results_dict.items():
        dz_norms = [jnp.linalg.norm(g[f"d{target}_dz"]) for g in grads_list]
        plt.plot(ls_grid, dz_norms, label=model_name, marker="o")
    plt.title(f"∥d{target}/dz∥ over ls")
    plt.xlabel("ls")
    plt.ylabel(f"Norm ∥d{target}/dz∥")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"d{target}_dz_vs_ls.png")
    plt.close()

    # 3. ∥d(target)/dls∥ norm
    plt.figure(figsize=(8, 6))
    for model_name, (_, grads_list, _) in results_dict.items():
        dls_norms = [
            jnp.linalg.norm(jnp.atleast_1d(g[f"d{target}_dls"])) for g in grads_list
        ]
        plt.plot(ls_grid, dls_norms, label=model_name, marker="o")
    plt.title(f"∥d{target}/dls∥ over ls")
    plt.xlabel("ls")
    plt.ylabel(f"Norm ∥d{target}/dls∥")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"d{target}_dls_vs_ls.png")
    plt.close()


def diff_per_loader(models_dict, s, s_train, kernel, save_dir, max_ls=50.0, bs=32):
    plot_dir = save_dir / "numerical_precision_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = random.key(17)
    surr_models = get_surr_decs(models_dict, save_dir)
    f_u_bar_mses = {m: [] for m in surr_models.keys()}
    f_mses = {m: [] for m in surr_models.keys()}
    kwargs = {"s": s_train}
    ls_grid = jnp.linspace(1.0, max_ls, 15)
    selected_ls_indices = [0, 5, 10, 14]
    K_uu_cond_num = []
    for ls_idx, ls_val in enumerate(ls_grid):
        priors = {"ls": dist.Delta(ls_val)}
        batch = gen_loader(kernel, s_train, priors, bs)(rng).__next__()
        K_su = kernel(s, s_train, 1.0, ls_val)
        f_u_bar = batch["f_u_bar"].squeeze()  # GT f_u_bar
        f = jnp.einsum("ij,bj->bi", K_su, f_u_bar)  # GT f from inducing points
        L_uu, K_uu = batch["L_uu"], batch["K_uu"]
        # Precompute SVD of K_su for residual projection
        U, S, Vh = jnp.linalg.svd(K_su, full_matrices=False)
        K_uu_cond_num.append(jnp.linalg.cond(K_uu))
        for m_name, surr_dec in surr_models.items():
            pred_f_u_bar = surr_dec(batch["z"], jnp.array([ls_val]), **kwargs).squeeze()
            if "inv" not in m_name:
                pred_f_u_bar = jit_trin_solve_func(
                    L_uu.T, jit_trin_solve_func(L_uu, pred_f_u_bar, True), False
                )
            pred_f = jnp.einsum("ij,bj->bi", K_su, pred_f_u_bar)

            if ls_idx in selected_ls_indices:
                residuals = f_u_bar - pred_f_u_bar
                f_residuals = jnp.einsum("ij,bj->bi", K_su, residuals)  # f - pred_f
                ls_folder = plot_dir / m_name / f"ls_{float(ls_val):.1f}"
                ls_folder.mkdir(parents=True, exist_ok=True)
                plot_residuals(residuals, f_residuals, m_name, ls_val, ls_folder)
                plot_energies(U, f_residuals, m_name, ls_val, ls_folder)

            mse_jit = optax.l2_loss(pred_f_u_bar, f_u_bar).mean()
            f_u_bar_mses[m_name].append(mse_jit)
            f_mses[m_name].append(optax.l2_loss(pred_f, f).mean())
    plot_K_uu_cond(K_uu_cond_num, plot_dir, ls_grid)
    plot_mses(f_u_bar_mses, ls_grid, plot_dir, f_mses)


def plot_energies(U, f_residuals, m_name, ls_val, ls_folder):
    proj = f_residuals @ U  # Shape: (B, U)
    proj_energy = jnp.mean(proj.T**2, axis=1)  # Mean energy per SVD mode
    plt.figure(figsize=(6, 4))
    plt.semilogy(proj_energy, marker="o")
    plt.title(f"Residual Energy in SVD basis\nModel: {m_name}, ls={ls_val:.1f}")
    plt.xlabel("SVD Mode")
    plt.ylabel("Mean Energy")
    plt.tight_layout()
    plt.savefig(ls_folder / "svd_residual_energy.png")
    plt.close()


def plot_K_uu_cond(K_uu_cond, plot_dir, ls_grid):
    plt.figure(figsize=(6, 4))
    plt.plot(ls_grid, K_uu_cond, marker="o")
    plt.title("K_uu condition number over lengthscale")
    plt.xlabel("ls")
    plt.ylabel(r"Condition Number $\kappa = \frac{\sigma_{max}}{\sigma_{min}}$")
    plt.tight_layout()
    plt.savefig(plot_dir / "K_uu_cond.png")
    plt.close()


def plot_mses(f_u_bar_mses, ls_grid, plot_dir, f_mses):
    plt.figure(figsize=(8, 6))
    for model_name, mse_vals in f_u_bar_mses.items():
        plt.plot(ls_grid, mse_vals, label=model_name, marker="o")
    plt.title("Mean MSE vs ls: f_u_bar")
    plt.xlabel("ls")
    plt.ylabel("Mean MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "f_u_bar_mse_vs_ls.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    for model_name, mse_vals in f_mses.items():
        plt.plot(ls_grid, mse_vals, label=model_name, marker="o")
    plt.title("Mean MSE vs ls:  full inducing GP")
    plt.xlabel("ls")
    plt.ylabel("Mean MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "f_mse_vs_ls.png")
    plt.close()


def plot_residuals(residuals, f_residuals, m_name, ls_val, ls_folder):
    for j in range(residuals.shape[0]):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(residuals[j], marker="o", linewidth=0.3)
        plt.title(
            "Residual (f_u_bar - model_res)\n"
            f"Model: {m_name}, ls={ls_val:.1f}, Sample {j}\n"
            f"Med: {jnp.median(residuals[j]):.3f}, Mean: {jnp.mean(residuals[j]):.3f}, "
            f"Std: {jnp.std(residuals[j]):.3f}"
        )
        plt.xlabel("Dimension")
        plt.ylabel("Residual")
        plt.subplot(2, 1, 2)
        plt.plot(f_residuals[j], marker="o", color="tab:orange", linewidth=0.5)
        plt.title(
            "F-space Residual (K_su@f_u_bar - K_su@model_res)\n"
            f"Mean: {jnp.mean(f_residuals[j]):.3f}, Std: {jnp.std(f_residuals[j]):.3f}"
        )
        plt.xlabel("Dimension")
        plt.ylabel("F Residual")
        plt.tight_layout()
        plt.savefig(ls_folder / f"sample_{j}.png")
        plt.close()


@partial(jit, static_argnames=["lower"])
def jit_trin_solve_func(L_T, z, lower=False):
    s_trin = partial(solve_triangular, lower=lower)
    return vmap(s_trin, in_axes=(None, 0))(L_T, z)


def gen_loader(kernel: Callable, s_train: Array, priors: dict, batch_size: int):
    jitter = 5e-4 * jnp.eye(s_train.shape[0])
    kernel_jit = jit(lambda s, var, ls: kernel(s, s, var, ls) + jitter)
    f_jit = jit(lambda L, z: jnp.einsum("ij,bj->bi", L, z))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z = random.split(rng_data, 3)
            var = 1.0
            ls = priors["ls"].sample(rng_ls)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s_train.shape[0]))
            K_uu = kernel_jit(s_train, var, ls)
            L_uu = jnp.linalg.cholesky(K_uu)
            f_u = f_jit(L_uu, z)
            f_u_bar = jit_trin_solve_func(L_uu.T, z, False)

            yield {
                "s": s_train,
                "K_uu": K_uu,
                "L_uu": L_uu,
                "f_u_bar": f_u_bar,
                "f_u": f_u,
                "z": z,
                "conditionals": jnp.array([ls]),
            }

    return dataloader


def get_surr_decs(models_dict, save_dir):
    lr_schedule = cosine_annealing_lr(200_000, 5.0e-3)
    optimizer = optax.chain(optax.clip_by_global_norm(3.0), optax.yogi(lr_schedule))
    ckptr = PyTreeCheckpointer()
    surr_models = {}
    for model_name, nn_model in models_dict.items():
        if nn_model is None:
            continue
        ckpt = ckptr.restore(((save_dir / f"{model_name}") / "model.ckpt").absolute())
        state = TrainState.create(
            apply_fn=nn_model.apply,
            tx=optimizer,
            params=ckpt["state"]["params"],
            kwargs=ckpt["state"]["kwargs"],
        )
        surr_models[model_name] = generate_surrogate_decoder(state, nn_model)
    return surr_models
