import sys

sys.path.append("benchmarks/vae")
import pickle
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Union

import arviz as az
import flax.linen as nn
import jax.numpy as jnp
import numpyro
import optax
import pandas as pd
from jax import Array, jit, random, value_and_grad
from jax.scipy.linalg import solve_triangular
from jax.scipy.stats import poisson
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from omegaconf import DictConfig
from reproduce_paper.deep_rv_plots import plot_posterior_predictive_comparisons
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sps.kernels import matern_1_2
from utils.plot_utils import plot_infer_trace

import wandb
from dl4bi.core.mlp import MLP
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import TrainState, cosine_annealing_lr, evaluate, save_ckpt, train
from dl4bi.vae import KernelBiasTransformerDeepRV
from dl4bi.vae.train_utils import generate_surrogate_decoder


def main(seed=23):
    save_dir = Path("results/multi_location/")
    save_dir.mkdir(parents=True, exist_ok=True)
    num_infer_locations = [512, 1024, 2048]
    max_s = 50.0
    rng = random.key(seed)
    kernel = matern_1_2
    rng_train, rng_test, rng_infer, rng_s, rng_obs = random.split(rng, 5)
    result, y_hats, all_samples = [], [], []
    default_steps = 2_000_000
    s_per_loc = [
        random.uniform(random.fold_in(rng_s, i), (n, 2), maxval=max_s)
        for i, n in enumerate(num_infer_locations)
    ]
    y_per_loc = [
        gen_y_obs(random.fold_in(rng_obs, i), s, 20.0, matern_1_2)
        for i, s in enumerate(s_per_loc)
    ]
    D = 128
    L = max(num_infer_locations)
    models = {
        "Baseline_GP": None,
        "Inducing Points Large": None,
        "DeepRV + trans kAttn": KernelBiasTransformerDeepRV(
            num_blks=4,
            dim=D,
            s_embed=RFFEmbed(num_features=512),
            head=MLP([D * 4, D, D // 2, 1]),
            max_locations=L,
        ),
    }
    for model_name, nn_model in models.items():
        model_path = save_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        train_time, eval_mse, surrogate_decoder = None, None, None
        max_lr, bs, train_steps = None, None, None
        if nn_model is not None:
            optimizer, max_lr, bs, train_steps = gen_train_params(default_steps)
            pr = {"ls": dist.Uniform(1.0, max_s)}
            loader = gen_dataloader(L, pr, kernel, max_s, bs, min_train_locs=512)
            wandb.init(
                config={
                    "model_name": model_name,
                    "grid_size": L,
                    "max_lr": max_lr,
                    "batch_size": bs,
                    "kernel": kernel.__name__,
                },
                mode="online",
                name=f"{model_name}_{kernel.__name__}",
                project="deep_rv_multiple_s",
                reinit=True,
            )
            train_time, eval_mse, surrogate_decoder = (
                surrogate_model_train(
                    rng_train,
                    rng_test,
                    loader,
                    nn_model,
                    optimizer,
                    train_steps,
                    model_path,
                )
                if not (model_path / "model.ckpt").exists()
                else load_surr_model_results(nn_model, optimizer, model_path)
            )
            wandb.log({"train_time": train_time, "Test Norm MSE": eval_mse})
        for i, (s, y_obs) in enumerate(zip(s_per_loc, y_per_loc)):
            model_s_path = model_path / f"{s.shape[0]}"
            model_s_path.mkdir(parents=True, exist_ok=True)
            obs_mask = True
            priors = {"ls": dist.Uniform(1.0, max_s), "beta": dist.Normal()}
            if "Inducing" in model_name:
                infer_model, cond_names = inference_model_inducing_points(s, priors)
            else:
                infer_model, cond_names = inference_model(s, priors, model_name, L)
            if (model_s_path / "single_res.pkl").exists():
                with open(model_s_path / "hmc_samples.pkl", "rb") as out_file:
                    samples = pickle.load(out_file)
                with open(model_s_path / "hmc_pp.pkl", "rb") as out_file:
                    post = pickle.load(out_file)
                with open(model_s_path / "single_res.pkl", "rb") as out_file:
                    res = pickle.load(out_file)
                y_hats.append(post["obs"])
                samples = {k: it for k, it in samples.items() if k in cond_names}
                all_samples.append(samples)
                result.append(res)
                continue

            samples, mcmc, post, infer_time = hmc(
                rng_infer,
                infer_model,
                y_obs,
                obs_mask,
                model_s_path,
                surrogate_decoder,
            )
            ess = az.ess(mcmc, method="mean")
            plot_infer_trace(
                samples, mcmc, None, cond_names, model_s_path / "infer_trace.png"
            )
            y_hats.append(post["obs"])
            all_samples.append(samples)
            res = {
                "model_name": model_name,
                "max_lr": max_lr,
                "bs": bs,
                "train_steps": default_steps,
                "grid_size": s.shape[0],
                "train_time": train_time,
                "Test Norm MSE": eval_mse,
                "infer_time": infer_time,
                "total_time": infer_time
                if train_time is None
                else infer_time + train_time,
                "MSE(y, y_hat)": ((y_obs - post["obs"].mean(axis=0)) ** 2).mean(),
                "seed": seed,
            }
            res.update(
                {f"inferred {c} mean": samples[c].mean(axis=0) for c in cond_names}
            )
            res.update(
                {f"ESS {c}": ess[c].mean().item() if ess else None for c in cond_names}
            )
            distance = None
            if model_name != "Baseline_GP":
                gp_samples = all_samples[i]
                for var_name in cond_names:
                    distance = wasserstein_distance(
                        gp_samples[var_name], samples[var_name]
                    )
                    res[f"{var_name} wasserstein distance"] = distance
            res["MSE(y_hat_gp, y_hat)"] = jnp.mean(
                (y_hats[i].mean(axis=0) - post["obs"].mean(axis=0)) ** 2
            )
            res["LPD"] = jnp.mean(
                jnp.log(jnp.mean(poisson.pmf(y_obs, post["obs"]), axis=0) + 1e-12)
            )
            lower = jnp.quantile(post["obs"], 0.1, axis=0)
            upper = jnp.quantile(post["obs"], 0.9, axis=0)
            res["coverage_80"] = jnp.mean((y_obs >= lower) & (y_obs <= upper))
            with open(model_s_path / "single_res.pkl", "wb") as out_file:
                pickle.dump(res, out_file)
            result.append(res)
    model_names = list(models.keys())
    for i in range(len(s_per_loc)):
        plot_posterior_predictive_comparisons(
            [samp for j, samp in enumerate(all_samples) if j % len(s_per_loc) == i],
            {},
            priors,
            model_names,
            cond_names,
            save_dir / f"comp_{s_per_loc[i].shape[0]}",
        )
    pd.DataFrame(result).to_csv(save_dir / "res.csv")


def hmc(
    rng: Array,
    model: Callable,
    y_obs: Array,
    obs_mask: Union[bool, Array],
    results_dir: Path,
    surrogate_decoder: Optional[Callable] = None,
):
    """runs HMC on given inference model and observed f"""
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
    # mcmc = MCMC(nuts, num_chains=1, num_samples=30, num_warmup=10)
    mcmc = MCMC(nuts, num_chains=2, num_samples=4_000, num_warmup=2_000)
    start = datetime.now()
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask, y=y_obs)
    infer_time = (datetime.now() - start).total_seconds()
    mcmc.print_summary()
    samples = mcmc.get_samples()
    post = Predictive(model, samples)(k2, surrogate_decoder=surrogate_decoder)
    post["infer_time"] = infer_time
    with open(results_dir / "hmc_samples.pkl", "wb") as out_file:
        pickle.dump(samples, out_file)
    with open(results_dir / "hmc_pp.pkl", "wb") as out_file:
        pickle.dump(post, out_file)
    samples = {k: it for k, it in samples.items() if k in ["ls", "beta"]}
    return samples, mcmc, post, infer_time


def surrogate_model_train(
    rng_train: Array,
    rng_test: Array,
    loader: Callable,
    model: nn.Module,
    optimizer,
    train_num_steps: int,
    results_dir: Path,
    valid_interval: int = 400_000,
    valid_steps: int = 10_000,
):
    start = datetime.now()
    state = train(
        rng_train,
        model,
        optimizer,
        masked_deep_rv_train_step,
        train_num_steps,
        loader,
        valid_step,
        valid_interval,
        valid_steps,
        loader,
        return_state="best",
        valid_monitor_metric="norm MSE",
    )
    save_ckpt(state, DictConfig({}), results_dir / "model.ckpt")
    train_time = (datetime.now() - start).total_seconds()
    eval_mse = evaluate(rng_test, state, valid_step, loader, valid_steps)["norm MSE"]
    with open(results_dir / "train.pkl", "wb") as ff:
        pickle.dump({"eval_mse": eval_mse, "train_time": train_time}, ff)
    return train_time, eval_mse, generate_surrogate_decoder(state, model)


@partial(jit, static_argnames=["max_s", "grid_size", "min_train_locs"])
def sample_uniform_s(rng, max_s, grid_size, min_train_locs):
    rng_loc, rng_s = random.split(rng)
    s = random.uniform(rng_s, (grid_size, 2), maxval=max_s)
    if min_train_locs < grid_size:
        num_locs = random.randint(
            rng_loc, shape=tuple(), minval=min_train_locs, maxval=grid_size + 1
        )
    return s, num_locs


def gen_dataloader(grid_size, priors, kernel, max_s, batch_size, min_train_locs=None):
    jitter = 5e-4 * jnp.eye(grid_size)
    min_train_locs = grid_size if min_train_locs is None else min_train_locs
    s_jit = partial(
        sample_uniform_s,
        max_s=max_s,
        grid_size=grid_size,
        min_train_locs=min_train_locs,
    )
    kernel_jit = jit(lambda s, var, ls: kernel(s, s, var, ls) + jitter)
    f_jit = jit(lambda K, z: jnp.einsum("ij,bj->bi", jnp.linalg.cholesky(K), z))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z, rng_samp = random.split(rng_data, 4)
            var = 1.0
            s, num_locs = s_jit(rng_samp)
            mask = jnp.repeat(
                (jnp.arange(grid_size) < num_locs)[None], batch_size, axis=0
            )
            ls = priors["ls"].sample(rng_ls)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, grid_size)) * mask
            K = kernel_jit(s, var, ls)
            f = f_jit(K, z)
            yield {
                "s": s,
                "f": f,
                "z": z,
                "mask": mask,
                "conditionals": jnp.array([ls]),
                "K": K,
            }

    return dataloader


@partial(jit, static_argnames=["var_idx"])
def masked_deep_rv_train_step(
    rng: Array,
    state,
    batch: dict,
    var_idx: Optional[int] = None,
):
    """DeepRV training step, MSE(f, f_hat).
    Can be normalized by variance to stabilize training, if
    variance is given as a conditional parameter.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.
        var_idx: the variance conditional index (if exists)

    Returns:
        `TrainState` with updated parameters, and the loss
    """

    def deep_rv_loss(params):
        conditionals, mask = batch["conditionals"], batch["mask"]
        var = conditionals[var_idx] if var_idx is not None else 1.0
        output: VAEOutput = state.apply_fn(
            {"params": params}, **batch, rngs={"extra": rng}
        )
        diff = (batch["f"].squeeze() - output.f_hat.squeeze()) * mask
        return (1 / var) * (jnp.sum(diff**2) / jnp.sum(mask))

    loss, grads = value_and_grad(deep_rv_loss)(state.params)
    return state.apply_gradients(grads=grads), loss


@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    mask = batch["mask"]
    diff = (batch["f"].squeeze() - output.f_hat.squeeze()) * mask
    return {"norm MSE": (jnp.sum(diff**2) / jnp.sum(mask))}


def gen_y_obs(rng: Array, s: Array, gt_ls: float, kernel: Callable):
    """generates a poisson observed data sample for inference"""
    rng_mu, rng_poiss = random.split(rng)
    var, ls, beta = 1.0, gt_ls, 1.0
    K = kernel(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
    mu = dist.MultivariateNormal(0.0, K).sample(rng_mu)
    lambda_ = jnp.exp(beta + mu)
    return dist.Poisson(rate=lambda_).sample(rng_poiss)


def gen_train_params(default_steps, default_bs=8):
    bs = default_bs
    max_lr = 1e-4
    train_steps = default_steps
    optimizer = optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.adamw(cosine_annealing_lr(train_steps, max_lr)),
    )
    return optimizer, max_lr, bs, train_steps


def inference_model(s: Array, priors: dict, model_name: str, max_infer_locs: int):
    """
    Builds a poisson likelihood inference model for GP and surrogate models
    """
    L_inf = s.shape[0]
    if "gMLP" in model_name and max_infer_locs - L_inf > 0:
        s = jnp.concatenate(
            [s, jnp.zeros((max_infer_locs - L_inf, s.shape[1]))], axis=0
        )
    L = s.shape[0]
    mask = jnp.repeat((jnp.arange(s.shape[0]) < L_inf)[None], 1, axis=0)
    surrogate_kwargs = {"s": s, "mask": mask}

    def poisson(surrogate_decoder=None, obs_mask=True, y=None):
        ls = numpyro.sample("ls", priors["ls"], sample_shape=())
        beta = numpyro.sample("beta", priors["beta"], sample_shape=())
        z = numpyro.sample("z", dist.Normal(), sample_shape=(1, L_inf))
        if L_inf < L:
            z = jnp.concatenate([z, jnp.zeros((1, L - L_inf))], axis=1)
        K = matern_1_2(s, s, 1.0, ls) + 5e-4 * jnp.eye(L)
        if surrogate_decoder:  # NOTE: whether to use a replacment for the GP
            surrogate_kwargs["K"] = K
            mu = numpyro.deterministic(
                "mu",
                surrogate_decoder(z, jnp.array([ls]), **surrogate_kwargs).squeeze()[
                    :L_inf
                ],
            )
        else:
            L_chol = jnp.linalg.cholesky(K)
            mu = numpyro.deterministic("mu", jnp.matmul(L_chol, z[0]))
        lambda_ = jnp.exp(beta + mu)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Poisson(rate=lambda_), obs=y)

    return poisson, ["ls", "beta"]


def inference_model_inducing_points(s: Array, priors: dict):
    """Builds a poisson likelihood inference model for inducing points"""
    n_pts = int(jnp.pow(s.shape[0], 2 / 3))
    kmeans = KMeans(n_clusters=n_pts, random_state=0)
    u = kmeans.fit(s).cluster_centers_

    def poisson_inducing(surrogate_decoder=None, obs_mask=True, y=None):
        var = 1.0
        ls = numpyro.sample("ls", priors["ls"], sample_shape=())
        beta = numpyro.sample("beta", priors["beta"], sample_shape=())
        K_su = matern_1_2(s, u, var, ls)
        z = numpyro.sample("z", dist.Normal(), sample_shape=(1, u.shape[0]))
        K_uu = matern_1_2(u, u, var, ls) + 5e-4 * jnp.eye(u.shape[0])
        L_uu = jnp.linalg.cholesky(K_uu)
        f_u_bar = solve_triangular(L_uu.T, z[0], lower=False)
        f = numpyro.deterministic("mu", K_su @ f_u_bar)
        lambda_ = jnp.exp(f + beta)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Poisson(lambda_), obs=y)

    return poisson_inducing, ["ls", "beta"]


class RFFEmbed(nn.Module):
    num_features: int = 128
    scale: float = 1.0
    include_original: bool = True
    head: Union[Callable, nn.Module] = lambda x: x

    @nn.compact
    def __call__(self, s: Array):
        D = s.shape[-1]
        W = self.param(
            "rff_W",
            lambda k: random.normal(k, shape=(self.num_features, D)) * self.scale,
        )
        b = self.param(
            "rff_b",
            lambda k: random.uniform(
                k, shape=(self.num_features,), minval=0.0, maxval=2 * jnp.pi
            ),
        )
        s_proj = jnp.dot(s, W.T) + b
        rff = jnp.sqrt(2.0 / self.num_features) * jnp.cos(s_proj)
        if self.include_original:
            s_encoded = jnp.concatenate([s, rff], axis=-1)
        else:
            s_encoded = rff
        return self.head(s_encoded)


def load_surr_model_results(model, optimizer, model_path: Path):
    from orbax.checkpoint import PyTreeCheckpointer

    ckptr = PyTreeCheckpointer()
    ckpt = ckptr.restore((model_path / "model.ckpt").absolute())
    state = TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=ckpt["state"]["params"],
        kwargs=ckpt["state"]["kwargs"],
    )
    surrogate_decoder = generate_surrogate_decoder(state, model)
    with open(model_path / "train.pkl", "rb") as ff:
        train_res = pickle.load(ff)
    eval_mse, train_time = train_res["eval_mse"], train_res["train_time"]
    return train_time, eval_mse, surrogate_decoder


if __name__ == "__main__":
    main()
