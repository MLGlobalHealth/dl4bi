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
    batch_size = data.batch_size
    num_ctx_steps = data.num_t - 1
    num_ctx_max = data.num_ctx_per_t.max
    num_test = data.num_test
    test_i = data.num_t - 1 if data.forecast else data.num_t // 2
    ctx_positions = tuple(i for i in range(data.num_t) if i != test_i)
    t = jnp.linspace(data.t.start, data.t.stop, data.num_steps, dtype=jnp.float32)
    use_irregular_layout = data.irregular_layout and not (
        for_plot and data.get("plot_regular_grid", True)
    )
    s_dims = (num_locations,) if use_irregular_layout else spatial_shape
    lo = jnp.asarray([axis["start"] for axis in data.s], dtype=jnp.float32)
    hi = jnp.asarray([axis["stop"] for axis in data.s], dtype=jnp.float32)
    base_flat = s_grid.reshape(-1, s_grid.shape[-1])

    repeat_batch = lambda v: jnp.repeat(v[None, ...], batch_size, axis=0)

    # Keep a custom batching path here because we sample GP values directly on
    # the queried points. Each yielded batch shares one sampled task and contains
    # multiple independent draws from that task, matching the cheaper policy used
    # in gp.py without reverting to dense episode sampling.
    @jit
    def sample_batch(rng: jax.Array):
        rng_s, rng_t, rng_perm, rng_len, rng_episode = random.split(rng, 5)
        if use_irregular_layout:
            s_flat = sample_uniform_points(rng_s, lo, hi, num_locations)
        else:
            s_flat = base_flat
        ts = select_time_indices(rng_t, data.num_steps, data.num_t, data.random_t)
        t_window = t[ts]
        permute_idxs = sample_permute_idxs(
            rng_perm,
            data.num_t,
            num_locations,
            data.independent_t_masks,
        )
        inv_permute_idx = jnp.argsort(permute_idxs, axis=1)
        valid_lens_ctx = sample_valid_lens(
            rng_len,
            num_ctx_steps,
            data.num_ctx_per_t.min,
            num_ctx_max,
            data.independent_t_masks,
        )
        ctx_s, ctx_t, ctx_mask = [], [], []
        for ctx_i, pos in enumerate(ctx_positions):
            idx = permute_idxs[pos, :num_ctx_max]
            s_ctx_i = s_flat[idx]
            t_ctx_i = jnp.full((num_ctx_max, 1), t_window[pos], dtype=jnp.float32)
            mask_ctx_i = jnp.arange(num_ctx_max) < valid_lens_ctx[ctx_i]
            ctx_s += [s_ctx_i]
            ctx_t += [t_ctx_i]
            ctx_mask += [mask_ctx_i]
        s_ctx = jnp.concatenate(ctx_s, axis=0)
        t_ctx = jnp.concatenate(ctx_t, axis=0)
        mask_ctx = jnp.concatenate(ctx_mask, axis=0)
        test_idx = permute_idxs[test_i, :num_test]
        s_test = s_flat[test_idx]
        t_test = jnp.full((num_test, 1), t_window[test_i], dtype=jnp.float32)
        mask_test = jnp.ones((num_test,), dtype=bool)
        s_query = jnp.concatenate([s_ctx, s_test], axis=0)
        t_query = jnp.concatenate([t_ctx[:, 0], t_test[:, 0]], axis=0)
        mask_query = jnp.concatenate([mask_ctx, mask_test], axis=0)
        f_query = sample_observations_shared_task(
            rng_episode,
            s_query,
            t_query,
            mask_query,
            data,
            batch_size,
        )
        num_ctx = num_ctx_steps * num_ctx_max
        f_ctx = f_query[:, :num_ctx, None]
        f_test = f_query[:, num_ctx:, None]
        return SpatiotemporalBatch(
            x_ctx=None,
            s_ctx=repeat_batch(s_ctx),
            t_ctx=repeat_batch(t_ctx),
            f_ctx=f_ctx,
            mask_ctx=repeat_batch(mask_ctx),
            x_test=None,
            s_test=repeat_batch(s_test),
            t_test=repeat_batch(t_test),
            f_test=f_test,
            mask_test=repeat_batch(mask_test),
            inv_permute_idx=inv_permute_idx,
            s_dims=s_dims,
            forecast=data.forecast,
        )

    def dataloader(rng: jax.Array):
        while True:
            rng_batch, rng = random.split(rng)
            yield sample_batch(rng_batch)

    return dataloader


def build_grid(axes: list[dict]):
    coords = [
        jnp.linspace(axis["start"], axis["stop"], axis["num"], dtype=jnp.float32)
        for axis in axes
    ]
    return jnp.stack(jnp.meshgrid(*coords, indexing="ij"), axis=-1)


def sample_uniform_points(
    rng: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
    num_locations: int,
):
    return random.uniform(
        rng,
        (num_locations, lo.shape[0]),
        minval=lo,
        maxval=hi,
    )


def select_time_indices(
    rng: jax.Array,
    num_steps: int,
    num_t: int,
    random_t: bool,
):
    if num_t == num_steps:
        return jnp.arange(num_t, dtype=jnp.int32)
    if random_t:
        return jnp.sort(random.choice(rng, num_steps, (num_t,), replace=False))
    last_t = random.randint(rng, (), num_t, num_steps)
    delta = jnp.arange(num_t - 1, -1, -1, dtype=jnp.int32)
    return last_t - delta


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
    valid_len = random.randint(rng, (), num_ctx_min, num_ctx_max)
    return jnp.full((num_ctx_steps,), valid_len, dtype=jnp.int32)


def sample_observations_shared_task(
    rng: jax.Array,
    s: jax.Array,
    t: jax.Array,
    mask: jax.Array,
    data: DictConfig,
    batch_size: int,
):
    rng_k, rng_m, rng_z, rng_eps = random.split(rng, 4)
    num_points = s.shape[0]
    kernel = sample_kernel_hyperparams(rng_k, data)
    mean = sample_mean_hyperparams(rng_m, data)
    m = mean_function_points(s, t, mean, data)
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
    valid = mask[:, None] & mask[None, :]
    eye = jnp.eye(num_points, dtype=jnp.float32)
    K = jnp.where(valid, K, 0.0)
    K += eye * (kernel["jitter"] + (~mask).astype(jnp.float32))
    chol = jnp.linalg.cholesky(K)
    z = random.normal(rng_z, (batch_size, num_points), dtype=jnp.float32)
    f = jnp.where(mask[None, :], m[None, :], 0.0) + z @ chol.T
    f += kernel["obs_noise"] * random.normal(rng_eps, f.shape, dtype=jnp.float32) * mask
    return jnp.where(mask[None, :], f, 0.0)


def sample_kernel_hyperparams(rng: jax.Array, data: DictConfig):
    rng_var, rng_ls, rng_a, rng_alpha, rng_b, rng_nu, rng_gamma, rng_noise = (
        random.split(rng, 8)
    )
    value_scale = jnp.asarray(data.get("value_scale", 1.0), dtype=jnp.float32)
    var = (
        sample_log_interval(
            rng_var,
            data.var_min,
            data.var_max,
        )
        * value_scale**2
    )
    ls_space = sample_log_interval(
        rng_ls,
        data.ls_space_min,
        data.ls_space_max,
    )
    a = sample_log_interval(
        rng_a,
        data.a_min,
        data.a_max,
    )
    alpha = sample_interval(
        rng_alpha,
        data.alpha_min,
        data.alpha_max,
    )
    b = sample_interval(
        rng_b,
        data.b_min,
        data.b_max,
    )
    nu = sample_interval(
        rng_nu,
        data.nu_min,
        data.nu_max,
    )
    gamma = sample_interval(
        rng_gamma,
        data.get("gamma_min", 1.0),
        data.get("gamma_max", 1.0),
    )
    obs_noise = (
        sample_log_interval(
            rng_noise,
            data.obs_noise_min,
            data.obs_noise_max,
        )
        * value_scale
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


def sample_mean_hyperparams(rng: jax.Array, data: DictConfig):
    (
        rng_bias,
        rng_terrain_weight,
        rng_clock,
        rng_harm_weight,
        rng_inter,
        rng_trend,
        rng_move_amp,
    ) = random.split(rng, 7)
    mean_offset = jnp.asarray(data.get("mean_offset", 0.0), dtype=jnp.float32)
    value_scale = jnp.asarray(data.get("value_scale", 1.0), dtype=jnp.float32)
    mean = data.mean
    return {
        "offset": mean_offset,
        "scale": value_scale,
        "bias": sample_interval(rng_bias, mean.bias_min, mean.bias_max),
        "terrain_weight": sample_interval(
            rng_terrain_weight,
            mean.terrain_weight_min,
            mean.terrain_weight_max,
        ),
        "clock_weights": sample_interval(
            rng_clock,
            mean.clock_weights_min,
            mean.clock_weights_max,
        ),
        "harmonic_weight": sample_interval(
            rng_harm_weight,
            mean.harmonic_weight_min,
            mean.harmonic_weight_max,
        ),
        "interaction_weight": sample_interval(
            rng_inter,
            mean.interaction_weight_min,
            mean.interaction_weight_max,
        ),
        "trend_weight": sample_interval(
            rng_trend,
            mean.trend_weight_min,
            mean.trend_weight_max,
        ),
        "moving_amp": sample_interval(
            rng_move_amp,
            mean.moving_amp_min,
            mean.moving_amp_max,
        ),
    }


def sample_interval(rng: jax.Array, low, high):
    low = jnp.asarray(low, dtype=jnp.float32)
    high = jnp.asarray(high, dtype=jnp.float32)
    return low + random.uniform(rng, low.shape, dtype=jnp.float32) * (high - low)


def sample_log_interval(rng: jax.Array, low, high):
    low = jnp.asarray(low, dtype=jnp.float32)
    high = jnp.asarray(high, dtype=jnp.float32)
    log_low = jnp.log(low)
    log_high = jnp.log(high)
    return jnp.exp(
        log_low
        + random.uniform(rng, low.shape, dtype=jnp.float32) * (log_high - log_low)
    )


def terrain_from_config(s: jax.Array, shared: DictConfig):
    terrain_cfg = shared.terrain
    centers = jnp.asarray(terrain_cfg.centers, dtype=jnp.float32)
    widths = jnp.asarray(terrain_cfg.widths, dtype=jnp.float32)
    amps = jnp.asarray(terrain_cfg.amps, dtype=jnp.float32)
    diff = (s[:, None, :] - centers[None, :, :]) / widths[None, :, :]
    terrain = jnp.sum(amps[None, :] * jnp.exp(-0.5 * jnp.sum(diff**2, axis=-1)), axis=1)
    phase = jnp.asarray(terrain_cfg.phase_pi, dtype=jnp.float32) * jnp.pi
    ripple_amp = jnp.asarray(terrain_cfg.ripple_amp, dtype=jnp.float32)
    ripple_x_freq_pi = jnp.asarray(terrain_cfg.ripple_x_freq_pi, dtype=jnp.float32)
    ripple_y_freq_pi = jnp.asarray(terrain_cfg.ripple_y_freq_pi, dtype=jnp.float32)
    ripple_phase_mix = jnp.asarray(terrain_cfg.ripple_phase_mix, dtype=jnp.float32)
    terrain += (
        ripple_amp
        * jnp.sin(ripple_x_freq_pi * jnp.pi * s[:, 0] + phase)
        * jnp.cos(ripple_y_freq_pi * jnp.pi * s[:, 1] + ripple_phase_mix * phase)
    )
    return standardize_feature(terrain)


def standardize_feature(x: jax.Array, eps: float = 1e-6):
    return (x - x.mean()) / (x.std() + eps)


def mean_function_points(
    s: jax.Array,
    t: jax.Array,
    params: dict,
    data: DictConfig,
):
    """Evaluate the deterministic mean function at pointwise spatiotemporal inputs."""
    shared = data.shared
    terrain = terrain_from_config(s, shared)
    clock_sin = jnp.sin(2 * jnp.pi * t)
    clock_cos = jnp.cos(2 * jnp.pi * t)
    moving_center0 = jnp.asarray(shared.moving.center0, dtype=jnp.float32)
    moving_velocity = jnp.asarray(shared.moving.velocity, dtype=jnp.float32)
    moving_width = jnp.asarray(shared.moving.width, dtype=jnp.float32)
    moving_center = (
        moving_center0[None, :] + (t[:, None] - 0.5) * moving_velocity[None, :]
    )
    diff = (s - moving_center) / moving_width[None, :]
    moving = params["moving_amp"] * jnp.exp(-0.5 * jnp.sum(diff**2, axis=-1))
    harmonic_phase = jnp.asarray(shared.harmonic_phase_pi, dtype=jnp.float32) * jnp.pi
    return params["offset"] + params["scale"] * (
        params["bias"]
        + params["terrain_weight"] * terrain
        + params["clock_weights"][0] * clock_sin
        + params["clock_weights"][1] * clock_cos
        + params["harmonic_weight"] * jnp.sin(4 * jnp.pi * t + harmonic_phase)
        + params["interaction_weight"] * terrain * clock_sin
        + params["trend_weight"] * (t - 0.5)
        + moving
    )


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
