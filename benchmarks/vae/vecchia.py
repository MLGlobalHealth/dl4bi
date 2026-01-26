import pickle
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Tuple

import jax.numpy as jnp
import optax
import pandas as pd
from jax import Array, jit, random, vmap
from jax.lax import scan, top_k
from sps.kernels import l2_dist, matern_1_2, matern_3_2, matern_5_2, rbf

import wandb
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, evaluate, train
from dl4bi.vae.deep_rv import gMLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step, generate_surrogate_decoder


def main(seed: int):
    wandb.init(mode="disabled")
    rng = random.key(seed)
    n_samples = 1000
    n_trials_per_kernel = 5
    n_sqrt = 48
    train_num_steps = 300_000
    batch_size = 32
    kernels = [rbf, matern_1_2, matern_3_2, matern_5_2]
    total_samples = n_samples * len(kernels) * n_trials_per_kernel
    res = []
    zs = jnp.zeros((total_samples, n_sqrt**2))
    gp_samples = jnp.zeros((total_samples, n_sqrt**2))
    for kernel_num, kernel in enumerate(kernels):
        for trial in range(n_trials_per_kernel):
            rng, rng_train, rng_s, rng_eval = random.split(rng, 4)
            min_l2_dist = -1
            while min_l2_dist < 0.05:  # NOTE: prevents overlapping points
                s = random.uniform(
                    rng_s, minval=0.0, maxval=100.0, shape=(n_sqrt**2, 2)
                )
                min_l2_dist = jnp.min(l2_dist(s, s) + jnp.eye(s.shape[0]))
                rng_s, _ = random.split(rng_s)

            lr_schedule = cosine_annealing_lr(train_num_steps, 2.0e-3)
            optimizer = optax.chain(
                optax.clip_by_global_norm(3.0),
                optax.adamw(lr_schedule, weight_decay=1e-2),
            )
            nn_model = gMLPDeepRV(num_blks=2)
            loader = gen_gp_train_loader(s, kernel, batch_size)
            eval_mse = 1
            # NOTE: a single RBF seed (third, trial=2) didn't converge well eval_mse=0.011, then we re-run
            # NOTE: only needed for 1 out of 20 training runs
            while eval_mse > 1e-2:
                state = train(
                    rng=rng_train,
                    model=nn_model,
                    optimizer=optimizer,
                    train_step=deep_rv_train_step,
                    train_num_steps=train_num_steps,
                    train_dataloader=loader,
                    valid_step=valid_step,
                    valid_interval=25_000,
                    valid_num_steps=5_000,
                    valid_dataloader=loader,
                    return_state="best",
                    valid_monitor_metric="norm MSE",
                )
                eval_mse = evaluate(rng_eval, state, valid_step, loader, 5_000)[
                    "norm MSE"
                ]
            surrogate_decoder = generate_surrogate_decoder(state, nn_model)
            m = n_sqrt * 2
            models = {
                "gp": gp_model_fn(s, kernel),
                "vecchia": vecchia_model_fn(s, m, kernel),
                "DeepRV": surrogate_model_fn(s, surrogate_decoder),
            }

            for i in range(n_samples + 1):
                rng, rng_ls, rng_z = random.split(rng, 3)
                var = 1.0
                ls = random.uniform(rng_ls, minval=1.0, maxval=100.0)
                z = random.normal(rng_z, shape=(s.shape[0],))
                single_res = {
                    "kernel": kernel.__name__,
                    "trial": trial,
                    "sample": i,
                    "ls": ls,
                    "DeepRV_eval_mse": eval_mse,
                }
                gp_sample = None
                for model_n, model_fn in models.items():
                    start = datetime.now()
                    sample = model_fn(var, ls, z)
                    sample_time = (datetime.now() - start).total_seconds()
                    single_res[f"time_{model_n}"] = sample_time
                    if model_n == "gp":
                        gp_sample = sample
                    else:
                        mse_w_gp = jnp.mean((gp_sample - sample) ** 2)
                        single_res[f"mse_{model_n}_w_gp"] = mse_w_gp
                # NOTE: remove the first sampling which is longer due to the jit of functions
                if i > 0:
                    total_idx = (
                        kernel_num * (n_trials_per_kernel * n_samples)
                        + trial * n_samples
                        + (i - 1)
                    )
                    zs = zs.at[total_idx].set(z)
                    gp_samples = gp_samples.at[total_idx].set(gp_sample)
                    res.append(single_res)
    res_dir = Path("results/vecchia")
    res_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res).to_csv(res_dir / "res.csv")
    with open(res_dir / "zs.pkl", "wb") as ff:
        pickle.dump(zs, ff)
    with open(res_dir / "gp_samples.pkl", "wb") as ff:
        pickle.dump(gp_samples, ff)


@partial(jit, static_argnames=["kernel_fn", "jitter"])
def gen_vecchia_var(
    s_ordered: Array,
    ls: float,
    var: float,
    neighbors_padded: Array,  # (n, m_max) with -1 padding
    kernel_fn: Callable,
    jitter: float = 5e-4,
) -> Tuple[Array, Array]:
    """
    Construct Vecchia linear predictors and conditional variances (vectorized).

    Computes, for each location i with neighbor set C(i):
      a_i = K_{C(i)}^{-1} k_{i,C(i)}  (solve K_{C(i)} a_i = k_{i,C(i)})
      v_i = var - k_{i,C(i)}^T a_i

    Padding: `neighbors_padded` uses `-1` to denote padded entries. Masked/padded entries are
    zeroed in outputs.

    Args:
      s_ordered: (n, d) ordered spatial locations.
      ls: lengthscale.
      var: process variance.
      neighbors_padded: int array (n, m_max) with neighbor indices or -1 for padding.
      kernel_fn: callable kernel(s1, s2, var, ls) -> (n1, n2) kernel matrix.
      jitter: numerical stability term

    Returns:
      A_padded: (n, m_max) Vecchia weights (zeros where padded).
      v: (n,) conditional variances
    """
    n = s_ordered.shape[0]
    # NOTE: compute full kernel matrix once
    full_kernel = kernel_fn(s_ordered, s_ordered, var, ls) + jitter * jnp.eye(n)
    # NOTE: precompute mask and safe indices for padded neighbors
    mask = neighbors_padded != -1  # (n, m_max)
    safe_indices = jnp.where(mask, neighbors_padded, 0)  # (n, m_max)

    @jit
    def process_location(i, safe_C_i, valid_mask):
        K_C = full_kernel[safe_C_i[:, None], safe_C_i[None, :]]  # (m_max, m_max)
        k_iC = full_kernel[i, safe_C_i]  # (m_max,)
        valid_mask_2d = valid_mask[:, None] & valid_mask[None, :]  # (m_max, m_max)
        K_C_safe = jnp.where(valid_mask_2d, K_C, 0.0)
        K_C_safe = K_C_safe + jnp.diag(~valid_mask * 1.0)
        k_iC_safe = jnp.where(valid_mask, k_iC, 0.0)
        a_i_full = jnp.linalg.solve(K_C_safe, k_iC_safe)  # (m_max,)
        a_i = jnp.where(valid_mask, a_i_full, 0.0)
        k_iC_masked = jnp.where(valid_mask, k_iC, 0.0)
        v_i = var - jnp.dot(k_iC_masked, a_i)
        return a_i, v_i

    # NOTE: vectorize over all locations using vmap
    A_padded, v = vmap(process_location, in_axes=(0, 0, 0))(
        jnp.arange(n), safe_indices, mask
    )

    return A_padded, v


@jit
def get_mu(C_padded: Array, A_padded: Array, v: Array, z: Array) -> Array:
    """
    Compute Vecchia-sampled f (GP approximation) sequentially using `scan`.

    Recurrence (for ordered locations i=0..n-1):
      f_i = a_i^T f_{C(i)} + sqrt(max(v_i,0)) * z_i

    Args:
      C_padded: (n, m) int array with neighbor indices or -1 for padding.
      A_padded: (n, m) float array of weights (zeros where padded).
      v: (n,) conditional variances
      z: (n,) standard normal draws.
    Returns:
      f: (n,) the Vecchia-sampled GP approximation.
    """
    n = C_padded.shape[0]
    mask = (C_padded != -1).astype(jnp.float32)  # (n, m)
    safe_indices = jnp.where(C_padded != -1, C_padded, 0)  # (n, m)

    def scan_fn(f, t):
        f_nei = f[safe_indices[t]] * mask[t]  # (m,)
        mu_t = jnp.dot(A_padded[t], f_nei)
        f_t = mu_t + jnp.sqrt(jnp.maximum(v[t], 0.0)) * z[t]
        updated_f = f.at[t].set(f_t)
        return updated_f, f_t

    f_init = jnp.zeros((n,), dtype=jnp.float32)
    f_final, _ = scan(scan_fn, f_init, jnp.arange(n))

    return f_final


@partial(jit, static_argnames=("m",))
def nearest_neighbors(s_ordered: Array, m: int) -> Array:
    """
    Compute nearest *previous* neighbors for Vecchia ordering.

    For each t in 0..n-1 select up to m previous indices j < t with smallest distance:
      C(t) = argmin_{j<t} dist(s_t, s_j) (take up to m)
    Padded entries use -1.

    Args:
      s_ordered: (n, d) ordered locations.
      m: maximum number of neighbors to return per location.

    Returns:
      neighbors_padded: (n, m) int32 array of neighbor indices with -1 padding.
    """
    n = s_ordered.shape[0]
    D: Array = l2_dist(s_ordered, s_ordered)  # (n, n)
    idx_all = jnp.arange(n)

    def body(carry, t):
        prev_mask = idx_all < t
        row = jnp.where(prev_mask, D[t, :], jnp.inf)  # (n,)
        _, inds = top_k(-row, k=m)  # inds: (m,)
        dsel = row[inds]  # (m,)
        order = jnp.argsort(dsel)  # (m,)
        inds = inds[order]
        valid_k = jnp.arange(m) < jnp.minimum(t, m)
        inds = jnp.where(valid_k, inds, -jnp.ones_like(inds))
        return carry, inds.astype(jnp.int32)

    _, neighbors = scan(body, None, jnp.arange(n))
    return neighbors


def vecchia_model_fn(
    s: Array,
    m: int,
    kernel_fn: Callable,
    order_fn: Callable | None = None,
    neigh_fn: Callable = nearest_neighbors,
):
    """
    Wrap a Vecchia approximation into the experiment's unified sampling API.

    Args:
      s: (N, d) spatial locations.
      m: maximum neighbors used by Vecchia.
      kernel_fn: kernel(s1, s2, var, ls) callable.
      order_fn: optional ordering function `order_fn(s) -> (s_ordered, ..., perm_inv)`.
      neigh_fn: neighbor selection function; defaults to `nearest_neighbors`.

    Returns:
      Vecchia's GP approximaton f of shape (N,)
    """
    N = s.shape[0]
    s_ordered = s
    perm_inv = jnp.arange(N, dtype=jnp.int32)
    if order_fn is not None:
        s_ordered, _, perm_inv = order_fn(s)
    C_padded = neigh_fn(s_ordered, m)

    def vecchia_gp(var: float, ls: float, z: Array, **kwargs):
        A_padded, v = gen_vecchia_var(s_ordered, ls, var, C_padded, kernel_fn)
        f = get_mu(C_padded, A_padded, v, z)
        return f[perm_inv]

    return vecchia_gp


def gp_model_fn(s: Array, kernel_fn: Callable):
    """
    Wrap a GP sampler into the experiment's unified API.

    Args:
      s: (N, d) spatial locations.
      kernel_fn: kernel(s1, s2, var, ls) callable.

    Returns:
      GP realization f of shape (N,)
    """

    def gp(var: float, ls: float, z: Array, **kwargs):
        K = kernel_fn(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
        L_chol = jnp.linalg.cholesky(K)
        f = jnp.matmul(L_chol, z.squeeze())
        return f

    return gp


def surrogate_model_fn(s: Array, surrogate_decoder: Callable):
    """
    Wrap the trained surrogate decoder to match the experiment sampling API.

    Args:
      s: (N, d) spatial locations.
      surrogate_decoder: callable produced by training pipeline that accepts
                         (z_batch, ls_array, s=...) and returns a VAEOutput-like object.

    Returns:
      The model's GP approximation
    """
    model_kwargs = {"s": s}

    def surr_dec(var: float, ls: float, z: Array, **kwargs):
        return surrogate_decoder(z[None], jnp.array([ls]), **model_kwargs).squeeze()

    return surr_dec


@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}


def gen_gp_train_loader(s: Array, kernel: Callable, batch_size):
    N = s.shape[0]
    jitter = 5e-4 * jnp.eye(N)
    f_jit = jit(lambda L, z: jnp.einsum("ij,bj->bi", L, z))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z = random.split(rng_data, 3)
            var = 1.0
            ls = random.uniform(rng_ls, minval=1.0, maxval=100.0)
            z = random.normal(rng_z, shape=(batch_size, N))
            K = kernel(s, s, var, ls) + jitter
            L = jnp.linalg.cholesky(K)
            yield {"s": s, "z": z, "conditionals": jnp.array([ls]), "f": f_jit(L, z)}

    return dataloader


if __name__ == "__main__":
    main(seed=18)
