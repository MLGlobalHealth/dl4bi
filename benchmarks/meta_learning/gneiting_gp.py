#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import wandb
from hydra.utils import instantiate
from jax import jit, random
from matplotlib.colors import Normalize
from omegaconf import DictConfig, OmegaConf

from dl4bi.core.train import Callback, TrainState, evaluate, save_ckpt, train
from dl4bi.meta_learning import SGNP
from dl4bi.meta_learning.data.spatiotemporal import SpatiotemporalBatch
from dl4bi.meta_learning.utils import cfg_to_run_name


@hydra.main("configs/gneiting_gp", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,
    )
    print(OmegaConf.to_yaml(cfg))
    path = Path(f"results/{cfg.project}/{cfg.data.name}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.key(cfg.seed)
    rng_graph, rng_train, rng_test = random.split(rng, 3)
    train_dataloader = build_dataloader(cfg.data)
    callback_dataloader = build_dataloader(cfg.data, for_plot=True)
    optimizer = instantiate(cfg.optimizer)
    model = instantiate(cfg.model)
    if not cfg.data.random_t and isinstance(model, SGNP):
        batch = next(train_dataloader(rng_graph))
        model = instantiate(cfg.model, graph=model.build_graph(**batch))
    output_fn = model.output_fn
    model = model.copy(output_fn=lambda x: output_fn(x, min_std=0.05))
    clbk = Callback(plot, cfg.plot_interval)
    state = train(
        rng_train,
        model,
        optimizer,
        model.train_step,
        cfg.train_num_steps,
        train_dataloader,
        model.valid_step,
        cfg.valid_interval,
        cfg.valid_num_steps,
        train_dataloader,
        callbacks=[clbk],
        callback_dataloader=callback_dataloader,
        return_state="best",
    )
    metrics = evaluate(
        rng_test,
        state,
        model.valid_step,
        train_dataloader,
        cfg.test_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloader(data: DictConfig, for_plot: bool = False):
    if data.num_t > data.num_steps:
        raise ValueError("num_t must be <= num_steps.")
    s_grid = build_grid(data.s)
    spatial_shape = s_grid.shape[:-1]
    num_locations = s_grid.reshape(-1, s_grid.shape[-1]).shape[0]
    if data.num_test > num_locations:
        raise ValueError("num_test must be <= the number of spatial locations.")
    t = jnp.linspace(data.t.start, data.t.stop, data.num_steps, dtype=jnp.float32)

    def dataloader(rng: jax.Array):
        while True:
            rng_batch, rng = random.split(rng)
            samples = [
                sample_batch_element(
                    rng_i,
                    s_grid,
                    spatial_shape,
                    num_locations,
                    t,
                    data,
                    for_plot,
                )
                for rng_i in random.split(rng_batch, data.batch_size)
            ]
            yield stack_batch(samples)

    return dataloader


def build_grid(axes: list[dict]):
    coords = [
        jnp.linspace(axis["start"], axis["stop"], axis["num"], dtype=jnp.float32)
        for axis in axes
    ]
    return jnp.stack(jnp.meshgrid(*coords, indexing="ij"), axis=-1)


def sample_jittered_grid(
    rng: jax.Array,
    axes: list[dict],
    jitter_frac: float,
):
    base = build_grid(axes)
    noise = random.uniform(rng, base.shape, minval=-1.0, maxval=1.0)
    steps = []
    for axis in axes:
        num = axis["num"]
        if num <= 1:
            steps.append(0.0)
        else:
            steps.append((axis["stop"] - axis["start"]) / (num - 1))
    step = jnp.asarray(steps, dtype=jnp.float32)
    lo = jnp.asarray([axis["start"] for axis in axes], dtype=jnp.float32)
    hi = jnp.asarray([axis["stop"] for axis in axes], dtype=jnp.float32)
    jitter = noise * (jitter_frac * step)
    return jnp.clip(base + jitter, lo, hi).reshape(-1, len(axes))


def sample_batch_element(
    rng: jax.Array,
    s_grid: jax.Array,
    spatial_shape: tuple[int, ...],
    num_locations: int,
    t_full: jax.Array,
    data: DictConfig,
    for_plot: bool,
):
    rng_s, rng_t, rng_perm, rng_len, rng_mode, rng_episode = random.split(rng, 6)
    use_irregular_layout = data.irregular_layout and not (
        for_plot and data.get("plot_regular_grid", True)
    )
    if use_irregular_layout:
        s_flat = sample_jittered_grid(
            rng_s,
            data.s,
            jitter_frac=data.spatial_jitter_frac,
        )
        s_dims = (num_locations,)
    else:
        s_flat = s_grid.reshape(-1, s_grid.shape[-1])
        s_dims = spatial_shape
    ts = select_time_indices(rng_t, data.num_steps, data.num_t, data.random_t)
    t_window = t_full[ts]
    test_i = data.num_t - 1 if data.forecast else data.num_t // 2
    ctx_positions = [i for i in range(data.num_t) if i != test_i]
    independent_t_masks = bool(data.independent_t_masks)
    independent_t_masks_prob = data.get("independent_t_masks_prob")
    if independent_t_masks_prob is not None:
        independent_t_masks = bool(
            random.bernoulli(
                rng_mode,
                jnp.asarray(independent_t_masks_prob, dtype=jnp.float32),
            )
        )
    permute_idxs = sample_permute_idxs(
        rng_perm,
        data.num_t,
        num_locations,
        independent_t_masks,
    )
    inv_permute_idx = jnp.argsort(permute_idxs, axis=1)
    valid_lens_ctx = sample_valid_lens(
        rng_len,
        len(ctx_positions),
        data.num_ctx_per_t.min,
        data.num_ctx_per_t.max,
        independent_t_masks,
    )
    num_ctx_max = data.num_ctx_per_t.max
    num_test = data.num_test
    query_s = []
    query_t = []
    ctx_indices = []
    for i, pos in enumerate(ctx_positions):
        idx = permute_idxs[pos, :num_ctx_max]
        ctx_indices += [idx]
        valid_len = int(valid_lens_ctx[i])
        if valid_len > 0:
            query_s += [s_flat[idx[:valid_len]]]
            query_t += [jnp.full((valid_len,), t_window[pos], dtype=jnp.float32)]
    test_idx = permute_idxs[test_i, :num_test]
    query_s += [s_flat[test_idx]]
    query_t += [jnp.full((num_test,), t_window[test_i], dtype=jnp.float32)]
    s_query = jnp.concatenate(query_s, axis=0)
    t_query = jnp.concatenate(query_t, axis=0)
    f_query = sample_observations(rng_episode, s_query, t_query, data)
    s_ctx = jnp.zeros((len(ctx_positions), num_ctx_max, s_flat.shape[-1]), dtype=jnp.float32)
    t_ctx = jnp.zeros((len(ctx_positions), num_ctx_max, 1), dtype=jnp.float32)
    f_ctx = jnp.zeros((len(ctx_positions), num_ctx_max, 1), dtype=jnp.float32)
    mask_ctx = jnp.zeros((len(ctx_positions), num_ctx_max), dtype=bool)
    cursor = 0
    for i, pos in enumerate(ctx_positions):
        idx = ctx_indices[i]
        valid_len = int(valid_lens_ctx[i])
        s_ctx = s_ctx.at[i].set(s_flat[idx])
        t_ctx = t_ctx.at[i].set(
            jnp.full((num_ctx_max, 1), t_window[pos], dtype=jnp.float32)
        )
        if valid_len > 0:
            f_ctx = f_ctx.at[i, :valid_len].set(
                f_query[cursor : cursor + valid_len].reshape(valid_len, 1)
            )
            mask_ctx = mask_ctx.at[i, :valid_len].set(True)
        cursor += valid_len
    f_test = f_query[cursor : cursor + num_test].reshape(num_test, 1)
    return {
        "s_ctx": s_ctx.reshape(-1, s_flat.shape[-1]),
        "t_ctx": t_ctx.reshape(-1, 1),
        "f_ctx": f_ctx.reshape(-1, 1),
        "mask_ctx": mask_ctx.reshape(-1),
        "s_test": s_flat[test_idx],
        "t_test": jnp.full((num_test, 1), t_window[test_i], dtype=jnp.float32),
        "f_test": f_test,
        "mask_test": jnp.ones((num_test,), dtype=bool),
        "inv_permute_idx": inv_permute_idx,
        "s_dims": s_dims,
        "forecast": data.forecast,
    }


def stack_batch(samples: list[dict]):
    return SpatiotemporalBatch(
        x_ctx=None,
        s_ctx=jnp.stack([sample["s_ctx"] for sample in samples]),
        t_ctx=jnp.stack([sample["t_ctx"] for sample in samples]),
        f_ctx=jnp.stack([sample["f_ctx"] for sample in samples]),
        mask_ctx=jnp.stack([sample["mask_ctx"] for sample in samples]),
        x_test=None,
        s_test=jnp.stack([sample["s_test"] for sample in samples]),
        t_test=jnp.stack([sample["t_test"] for sample in samples]),
        f_test=jnp.stack([sample["f_test"] for sample in samples]),
        mask_test=jnp.stack([sample["mask_test"] for sample in samples]),
        inv_permute_idx=jnp.stack([sample["inv_permute_idx"] for sample in samples]),
        s_dims=samples[0]["s_dims"],
        forecast=samples[0]["forecast"],
    )


def select_time_indices(
    rng: jax.Array,
    num_steps: int,
    num_t: int,
    random_t: bool,
):
    if num_t == num_steps:
        return jnp.arange(num_t)
    if random_t:
        return jnp.sort(random.choice(rng, num_steps, (num_t,), replace=False))
    last_t = random.randint(rng, (), num_t, num_steps)
    return last_t - jnp.arange(num_t - 1, -1, -1)


def sample_permute_idxs(
    rng: jax.Array,
    num_t: int,
    num_locations: int,
    independent_t_masks: bool,
):
    if independent_t_masks:
        return jnp.stack(
            [random.permutation(rng_i, num_locations) for rng_i in random.split(rng, num_t)]
        )
    permute_idx = random.permutation(rng, num_locations)
    return jnp.repeat(permute_idx[None, :], num_t, axis=0)


def sample_valid_lens(
    rng: jax.Array,
    num_ctx_steps: int,
    num_ctx_min: int,
    num_ctx_max: int,
    independent_t_masks: bool,
):
    if num_ctx_max <= num_ctx_min:
        return jnp.full((num_ctx_steps,), num_ctx_min, dtype=jnp.int32)
    if independent_t_masks:
        return random.randint(rng, (num_ctx_steps,), num_ctx_min, num_ctx_max)
    valid_len = random.randint(rng, (1,), num_ctx_min, num_ctx_max)
    return jnp.repeat(valid_len, num_ctx_steps)


def sample_episode(
    rng: jax.Array,
    s: jax.Array,
    t: jax.Array,
    spatial_shape: tuple[int, ...],
    data: DictConfig,
):
    s_rep, t_rep = flatten_spacetime_inputs(s, t)
    f = sample_observations(rng, s_rep, t_rep, data)
    return f.reshape(t.shape[0], *spatial_shape, 1)


def sample_observations(
    rng: jax.Array,
    s: jax.Array,
    t: jax.Array,
    data: DictConfig,
):
    rng_k, rng_m, rng_z, rng_eps = random.split(rng, 4)
    kernel = sample_kernel_hyperparams(rng_k, data)
    mean = sample_mean_hyperparams(rng_m)
    m = representative_mean_points(s, t, mean)
    K = gneiting_covariance_points(
        s,
        t,
        s,
        t,
        kernel["var"],
        kernel["ls_space"],
        kernel["a"],
        kernel["alpha"],
        kernel["b"],
        kernel["nu"],
        kernel["gamma"],
    )
    K += kernel["jitter"] * jnp.eye(K.shape[0], dtype=jnp.float32)
    z = random.normal(rng_z, (K.shape[0],), dtype=jnp.float32)
    f = m + jnp.linalg.cholesky(K).astype(jnp.float32) @ z
    if float(kernel["obs_noise"]) > 0.0:
        f += kernel["obs_noise"] * random.normal(rng_eps, f.shape, dtype=jnp.float32)
    return f


def cfg_value(data: DictConfig, primary: str, fallback: str | None = None):
    if primary in data:
        return data[primary]
    if fallback is not None and fallback in data:
        return data[fallback]
    raise KeyError(f"Expected one of {primary!r} or {fallback!r} in config.")


def sample_kernel_hyperparams(rng: jax.Array, data: DictConfig):
    rng_var, rng_ls, rng_a, rng_alpha, rng_b, rng_nu, rng_gamma, rng_noise = (
        random.split(rng, 8)
    )
    var = sample_log_interval(
        rng_var,
        cfg_value(data, "var_min", "var"),
        cfg_value(data, "var_max", "var"),
    )
    ls_space = sample_log_interval(
        rng_ls,
        cfg_value(data, "ls_space_min", "ls_min"),
        cfg_value(data, "ls_space_max", "ls_max"),
    )
    a = sample_log_interval(
        rng_a,
        cfg_value(data, "a_min"),
        cfg_value(data, "a_max"),
    )
    alpha = sample_interval(
        rng_alpha,
        cfg_value(data, "alpha_min"),
        cfg_value(data, "alpha_max"),
    )
    b = sample_interval(
        rng_b,
        cfg_value(data, "b_min", "b"),
        cfg_value(data, "b_max", "b"),
    )
    nu = sample_interval(
        rng_nu,
        cfg_value(data, "nu_min"),
        cfg_value(data, "nu_max"),
    )
    gamma = sample_interval(
        rng_gamma,
        data.get("gamma_min", 1.0),
        data.get("gamma_max", 1.0),
    )
    obs_noise = sample_log_interval(
        rng_noise,
        cfg_value(data, "obs_noise_min", "obs_noise"),
        cfg_value(data, "obs_noise_max", "obs_noise"),
    )
    return {
        "var": jnp.asarray(var, dtype=jnp.float32),
        "ls_space": jnp.asarray(ls_space, dtype=jnp.float32),
        "a": jnp.asarray(a, dtype=jnp.float32),
        "alpha": jnp.asarray(alpha, dtype=jnp.float32),
        "b": jnp.asarray(b, dtype=jnp.float32),
        "nu": jnp.asarray(nu, dtype=jnp.float32),
        "gamma": jnp.asarray(gamma, dtype=jnp.float32),
        "obs_noise": jnp.asarray(obs_noise, dtype=jnp.float32),
        "jitter": jnp.asarray(data.jitter, dtype=jnp.float32),
    }


def sample_mean_hyperparams(rng: jax.Array):
    (
        rng_bias,
        rng_terrain_weight,
        rng_clock,
        rng_harm_weight,
        rng_harm_phase,
        rng_inter,
        rng_trend,
        rng_move_amp,
        rng_move_center,
        rng_move_vel,
        rng_move_width,
        rng_terrain_shape,
    ) = random.split(rng, 12)
    return {
        "bias": sample_interval(rng_bias, -0.3, 0.3),
        "terrain_weight": sample_interval(rng_terrain_weight, -0.7, 0.7),
        "clock_weights": sample_interval(
            rng_clock,
            jnp.array([-0.65, -0.65], dtype=jnp.float32),
            jnp.array([0.65, 0.65], dtype=jnp.float32),
        ),
        "harmonic_weight": sample_interval(rng_harm_weight, -0.25, 0.25),
        "harmonic_phase": sample_interval(rng_harm_phase, -jnp.pi, jnp.pi),
        "interaction_weight": sample_interval(rng_inter, -0.4, 0.4),
        "trend_weight": sample_interval(rng_trend, -0.3, 0.3),
        "moving_amp": sample_interval(rng_move_amp, -0.65, 0.65),
        "moving_center0": sample_interval(
            rng_move_center,
            jnp.array([-0.65, -0.65], dtype=jnp.float32),
            jnp.array([0.65, 0.65], dtype=jnp.float32),
        ),
        "moving_velocity": sample_interval(
            rng_move_vel,
            jnp.array([-0.6, -0.6], dtype=jnp.float32),
            jnp.array([0.6, 0.6], dtype=jnp.float32),
        ),
        "moving_width": sample_log_interval(
            rng_move_width,
            jnp.array([0.18, 0.18], dtype=jnp.float32),
            jnp.array([0.45, 0.45], dtype=jnp.float32),
        ),
        "terrain": sample_terrain_params(rng_terrain_shape),
    }


def sample_interval(rng: jax.Array, low, high):
    low = jnp.asarray(low, dtype=jnp.float32)
    high = jnp.asarray(high, dtype=jnp.float32)
    if jnp.allclose(low, high).item():
        return low
    return random.uniform(rng, low.shape, minval=low, maxval=high)


def sample_log_interval(rng: jax.Array, low, high):
    low = jnp.asarray(low, dtype=jnp.float32)
    high = jnp.asarray(high, dtype=jnp.float32)
    if jnp.allclose(low, high).item():
        return low
    return jnp.exp(random.uniform(rng, low.shape, minval=jnp.log(low), maxval=jnp.log(high)))


def sample_terrain_params(rng: jax.Array):
    rng_centers, rng_widths, rng_amps, rng_phase = random.split(rng, 4)
    num_bumps, dims = 3, 2
    return {
        "centers": sample_interval(
            rng_centers,
            -0.85 * jnp.ones((num_bumps, dims), dtype=jnp.float32),
            0.85 * jnp.ones((num_bumps, dims), dtype=jnp.float32),
        ),
        "widths": sample_log_interval(
            rng_widths,
            0.18 * jnp.ones((num_bumps, dims), dtype=jnp.float32),
            0.6 * jnp.ones((num_bumps, dims), dtype=jnp.float32),
        ),
        "amps": random.normal(rng_amps, (num_bumps,), dtype=jnp.float32),
        "phase": sample_interval(rng_phase, -jnp.pi, jnp.pi),
    }


def terrain_from_params(s: jax.Array, params: dict):
    centers = params["centers"]
    widths = params["widths"]
    amps = params["amps"]
    diff = (s[:, None, :] - centers[None, :, :]) / widths[None, :, :]
    terrain = jnp.sum(amps[None, :] * jnp.exp(-0.5 * jnp.sum(diff**2, axis=-1)), axis=1)
    phase = params["phase"]
    terrain += 0.35 * jnp.sin(jnp.pi * s[:, 0] + phase) * jnp.cos(
        0.8 * jnp.pi * s[:, 1] - 0.5 * phase
    )
    return standardize_feature(terrain)


def flatten_spacetime_inputs(s: jax.Array, t: jax.Array):
    T, L = t.shape[0], s.shape[0]
    s_rep = jnp.broadcast_to(s[None, :, :], (T, L, s.shape[-1])).reshape(-1, s.shape[-1])
    t_rep = jnp.broadcast_to(t[:, None], (T, L)).reshape(-1)
    return s_rep, t_rep


def standardize_feature(x: jax.Array, eps: float = 1e-6):
    return (x - x.mean()) / (x.std() + eps)


def representative_mean_points(
    s: jax.Array,
    t: jax.Array,
    params: dict,
):
    terrain = terrain_from_params(s, params["terrain"])
    clock_sin = jnp.sin(2 * jnp.pi * t)
    clock_cos = jnp.cos(2 * jnp.pi * t)
    moving_center = params["moving_center0"][None, :] + (t[:, None] - 0.5) * params[
        "moving_velocity"
    ][None, :]
    diff = (s - moving_center) / params["moving_width"][None, :]
    moving = params["moving_amp"] * jnp.exp(-0.5 * jnp.sum(diff**2, axis=-1))
    return (
        params["bias"]
        + params["terrain_weight"] * terrain
        + params["clock_weights"][0] * clock_sin
        + params["clock_weights"][1] * clock_cos
        + params["harmonic_weight"] * jnp.sin(4 * jnp.pi * t + params["harmonic_phase"])
        + params["interaction_weight"] * terrain * clock_sin
        + params["trend_weight"] * (t - 0.5)
        + moving
    )


@jit
def gneiting_covariance_matrix(
    s1: jax.Array,
    t1: jax.Array,
    s2: jax.Array,
    t2: jax.Array,
    var: jax.Array,
    ls_space: jax.Array,
    a: jax.Array,
    alpha: jax.Array,
    b: jax.Array,
    nu: jax.Array,
    gamma: jax.Array,
):
    """Gneiting (2002) non-separable space-time covariance on R^d x R."""
    L1, T1 = s1.shape[0], t1.shape[0]
    L2, T2 = s2.shape[0], t2.shape[0]
    scaled_sq = jnp.sum(
        ((s1[:, None, :] - s2[None, :, :]) / ls_space[None, None, :]) ** 2,
        axis=-1,
    )
    u = jnp.abs(t1[:, None] - t2[None, :])
    scaled_sq = scaled_sq[None, None, :, :]
    u = u[:, :, None, None]
    g = 1.0 + a * u ** (2 * alpha)
    K = var / (g**nu) * jnp.exp(-((scaled_sq / (g**b)) ** gamma))
    return K.transpose(0, 2, 1, 3).reshape(T1 * L1, T2 * L2)


@jit
def gneiting_covariance_points(
    s1: jax.Array,
    t1: jax.Array,
    s2: jax.Array,
    t2: jax.Array,
    var: jax.Array,
    ls_space: jax.Array,
    a: jax.Array,
    alpha: jax.Array,
    b: jax.Array,
    nu: jax.Array,
    gamma: jax.Array,
):
    scaled_sq = jnp.sum(
        ((s1[:, None, :] - s2[None, :, :]) / ls_space[None, None, :]) ** 2,
        axis=-1,
    )
    u = jnp.abs(t1[:, None] - t2[None, :])
    g = 1.0 + a * u ** (2 * alpha)
    return var / (g**nu) * jnp.exp(-((scaled_sq / (g**b)) ** gamma))


def plot(
    step: int,
    rng_step: int,
    state: TrainState,
    batch: SpatiotemporalBatch,
    extra: dict | None = None,
    **kwargs,
):
    """Log a regular-grid synthetic forecast similarly to ERA5."""
    if extra:
        kwargs = {**extra, **kwargs}
    rng_dropout, rng_extra = random.split(rng_step)
    output = state.apply_fn(
        {"params": state.params, **state.kwargs},
        **batch,
        rngs={"dropout": rng_dropout, "extra": rng_extra},
    )
    if isinstance(output, tuple):
        output, _ = output
    f_pred, f_std = output.mu, output.std
    f_min = min(batch.f_ctx.min(), batch.f_test.min(), f_pred.min())
    f_max = max(batch.f_ctx.max(), batch.f_test.max(), f_pred.max())
    norm = Normalize(f_min, f_max)
    norm_std = Normalize(f_std.min(), f_std.max())
    cmap = mpl.colormaps.get_cmap("Spectral_r")
    cmap.set_bad("grey")
    path = f"/tmp/gneiting_gp_{step}_{datetime.now().isoformat()}.png"
    fig = batch.plot_2d(
        f_pred,
        f_std,
        cmap=cmap,
        norm=norm,
        norm_std=norm_std,
        **kwargs,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    wandb.log({f"Step {step}": wandb.Image(path)})


if __name__ == "__main__":
    main()
