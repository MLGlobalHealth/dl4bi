"""Prior sweep for FIXED 2D-cond setup (Makkunda's gneiting kernel,
a=nu=1 fixed, 2D cond=(ls,alpha), YOGI optimiser).

Tests how robust FM-DeepRV training is to:
  --ls-prior {narrow, wide, beta_logscale, lognormal}
  --extra-cond {none, a, nu}   -- adds a 3rd cond dim with tight prior

Both gMLP and FM share the same loader/cond per run.

Run on clpc35:/home/scratch/setman/dl4bi/.venv.
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import optax
from dl4bi.core.train import TrainState, cosine_annealing_lr, evaluate, train
from dl4bi.vae import FlowMatchingDeepRV, FlowMatchingVectorField, gMLPDeepRV
from dl4bi.vae.train_utils import (
    deep_rv_train_step,
    flow_matching_train_step,
    flow_matching_valid_step,
)
from jax import Array, jit, random
from numpyro import distributions as dist
from numpyro.distributions.transforms import ParameterFreeTransform

import wandb
from dl4bi.core.model_output import VAEOutput


@jit
def gneiting_makkunda(
    s1: Array, t1: Array, s2: Array, t2: Array,
    var: float, ls: float, a: float, alpha: float, nu: float,
) -> Array:
    L1, T1 = s1.shape[0], t1.shape[0]
    L2, T2 = s2.shape[0], t2.shape[0]
    h2 = jnp.sum((s1[:, None, :] - s2[None, :, :]) ** 2, axis=-1)
    u  = jnp.abs(t1[:, None] - t2[None, :])
    h2 = h2[None, None, :, :]
    u  = u[:, :, None, None]
    g  = 1.0 + a * u ** (2 * alpha)
    K  = var / (g ** nu) * jnp.exp(-h2 / (ls ** 2 * g))
    return K.transpose(0, 2, 1, 3).reshape(T1 * L1, T2 * L2)


@jit
def gneiting_ours(
    s1: Array, t1: Array, s2: Array, t2: Array,
    var: float, ls_s: float, ls_t: float, alpha: float,
) -> Array:
    """Matern-1/2 (exponential) spatial decay; psi-form temporal coupling."""
    L1, T1 = s1.shape[0], t1.shape[0]
    L2, T2 = s2.shape[0], t2.shape[0]
    h2 = jnp.sum((s1[:, None, :] - s2[None, :, :]) ** 2, axis=-1)
    u  = jnp.abs(t1[:, None] - t2[None, :])
    h  = jnp.sqrt(jnp.maximum(h2, 1e-12))[None, None, :, :]
    u  = u[:, :, None, None] / ls_t
    psi = u ** (2 * alpha) + 1.0
    K  = var / psi * jnp.exp(-h / (ls_s * psi ** (alpha / 2)))
    return K.transpose(0, 2, 1, 3).reshape(T1 * L1, T2 * L2)


class LogScaleTransform(ParameterFreeTransform):
    domain = dist.constraints.real
    codomain = dist.constraints.positive
    event_dim = 0
    def __call__(self, x):
        return jnp.exp(x * jnp.log(100.0))
    def _inverse(self, y):
        return jnp.log(y) / jnp.log(100.0)
    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.log(100.0) + x * jnp.log(100.0)


def make_ls_prior(name):
    if name == "narrow":
        return dist.TransformedDistribution(
            dist.Uniform(jnp.log(1.0), jnp.log(100.0)),
            dist.transforms.ExpTransform())
    if name == "wide":
        return dist.TransformedDistribution(
            dist.Uniform(jnp.log(0.1), jnp.log(1000.0)),
            dist.transforms.ExpTransform())
    if name == "beta_logscale":
        return dist.TransformedDistribution(dist.Beta(4.0, 1.0), LogScaleTransform())
    if name == "lognormal":
        return dist.LogNormal(jnp.log(20.0), 1.0)
    raise ValueError(name)


def gen_loader(st, s_all, t_all, ls_prior_name, extra_cond, batch_size, jitter,
               kernel_form="makkunda"):
    TL = st.shape[0]
    f_jit = jit(lambda K, z: jnp.einsum("ij,bj->bi", jnp.linalg.cholesky(K), z))
    ls_prior = make_ls_prior(ls_prior_name)
    alpha_prior = dist.Uniform(0.1, 0.95)
    use_a  = extra_cond in ("a", "a+nu")
    use_nu = extra_cond in ("nu", "a+nu")
    use_lst = extra_cond in ("lst",)
    a_prior  = dist.Uniform(0.8, 1.2) if use_a  else None
    nu_prior = dist.Uniform(1.0, 2.0) if use_nu else None
    lst_prior = dist.Uniform(0.5, 2.0) if use_lst else None

    def dataloader(rng_data):
        while True:
            rng_data, *rngs = random.split(rng_data, 6)
            ls    = ls_prior.sample(rngs[0])
            alpha = alpha_prior.sample(rngs[1])
            a     = a_prior.sample(rngs[2])  if a_prior  is not None else 1.0
            nu    = nu_prior.sample(rngs[3]) if nu_prior is not None else 1.0
            ls_t  = lst_prior.sample(rngs[2]) if lst_prior is not None else 1.0
            if kernel_form == "makkunda":
                K = gneiting_makkunda(s_all, t_all, s_all, t_all, 1.0, ls, a, alpha, nu)
            else:
                K = gneiting_ours(s_all, t_all, s_all, t_all, 1.0, ls, ls_t, alpha)
            K = K + jitter * jnp.eye(TL)
            z = dist.Normal().sample(rngs[4], sample_shape=(batch_size, TL))
            f = f_jit(K, z)
            if extra_cond == "none":
                cond = jnp.array([ls, alpha])
            elif extra_cond == "a":
                cond = jnp.array([ls, alpha, a])
            elif extra_cond == "nu":
                cond = jnp.array([ls, alpha, nu])
            elif extra_cond == "a+nu":
                cond = jnp.array([ls, alpha, a, nu])
            elif extra_cond == "lst":
                cond = jnp.array([ls, alpha, ls_t])
            yield {"s": st, "f": f, "z": z, "conditionals": cond}
    return dataloader


@jit
def deep_rv_valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    return {"norm MSE": output.metrics(batch["f"], 1.0)["MSE"]}


def yogi_chain(lr_schedule):
    return optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.scale_by_yogi(),
        optax.add_decayed_weights(1e-2),
        optax.scale_by_schedule(lambda step: -lr_schedule(step)),
    )


def train_one(model_name, nn_model, train_step, valid_step, loader, train_steps, rng):
    wandb.init(config={"model_name": model_name}, mode="disabled", reinit=True)
    rng_t, rng_v, rng_e = random.split(rng, 3)
    flop_batch = next(loader(rng_t))
    rngs = {"params": rng_t, "extra": rng_v}
    kwargs = nn_model.init(rngs, **flop_batch)
    params = kwargs.pop("params")
    lr_schedule = cosine_annealing_lr(train_steps, 5e-3)
    optimizer = yogi_chain(lr_schedule)
    state = TrainState.create(apply_fn=nn_model.apply, params=params, kwargs=kwargs, tx=optimizer)
    t0 = time.perf_counter()
    state = train(
        rng_t, nn_model, optimizer, train_step, train_steps, loader,
        valid_step, train_steps, 500, loader,
        return_state="best", valid_monitor_metric="norm MSE",
    )
    wall = time.perf_counter() - t0
    eval_mse = evaluate(rng_e, state, valid_step, loader, 500)["norm MSE"]
    return wall, float(eval_mse)


def build_grid(grid_size, time_steps):
    xs = jnp.linspace(0.0, 100.0, grid_size)
    ys = jnp.linspace(0.0, 100.0, grid_size)
    s = jnp.stack(jnp.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    t = jnp.arange(time_steps, dtype=jnp.float32)
    T, L = time_steps, s.shape[0]
    s_exp = jnp.broadcast_to(s, (T, L, 2))
    t_exp = t[:, None, None] * jnp.ones((1, L, 1))
    st = jnp.concatenate([s_exp, t_exp], axis=-1).reshape(T * L, 3)
    return s, t, st


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ls-prior", choices=["narrow", "wide", "beta_logscale", "lognormal"], default="narrow")
    p.add_argument("--extra-cond", choices=["none", "a", "nu", "a+nu", "lst"], default="none")
    p.add_argument("--kernel-form", choices=["makkunda", "ours"], default="makkunda")
    p.add_argument("--num-blks", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-steps", type=int, default=30_000)
    p.add_argument("--grid-size", type=int, default=8)
    p.add_argument("--time-steps", type=int, default=3)
    p.add_argument("--jitter", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--tag", type=str, default="prior_sweep")
    args = p.parse_args()

    print(f"jax devices: {jax.devices()}")
    print(f"config: ls-prior={args.ls_prior} extra-cond={args.extra_cond} bs={args.batch_size} "
          f"steps={args.train_steps} grid={args.grid_size}x{args.grid_size}x{args.time_steps}")

    rng = random.key(args.seed)
    s, t, st = build_grid(args.grid_size, args.time_steps)
    print(f"grid: {st.shape[0]} field points")

    loader = gen_loader(st, s, t, args.ls_prior, args.extra_cond,
                        args.batch_size, args.jitter, kernel_form=args.kernel_form)

    rng, rng_g = random.split(rng)
    print("\n--- gMLP-DeepRV ---")
    gmlp = gMLPDeepRV(num_blks=args.num_blks)
    wall_g, mse_g = train_one("gMLP", gmlp, deep_rv_train_step, deep_rv_valid_step,
                              loader, args.train_steps, rng_g)
    print(f"  wall={wall_g:.0f}s eval_norm_mse={mse_g:.4f}")

    rng, rng_f = random.split(rng)
    print("\n--- FM-DeepRV ---")
    fm = FlowMatchingDeepRV(vf=FlowMatchingVectorField(num_blks=args.num_blks), n_steps=1)
    wall_f, mse_f = train_one("FM", fm, flow_matching_train_step, flow_matching_valid_step,
                              loader, args.train_steps, rng_f)
    print(f"  wall={wall_f:.0f}s eval_norm_mse={mse_f:.4f}")

    print(f"\n[RESULT tag={args.tag}] ls-prior={args.ls_prior} extra-cond={args.extra_cond} "
          f"gMLP_mse={mse_g:.4f} FM_mse={mse_f:.4f} gMLP_wall={wall_g:.0f}s FM_wall={wall_f:.0f}s")


if __name__ == "__main__":
    main()
