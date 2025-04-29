"""
Facilitates training of the disease mapping model.
Learns the map
(s, N, N+)_ctx, s_test -> distribution over logit(theta)_test
"""

from functools import partial
from inspect import getsourcefile
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import wandb
from hydra.utils import instantiate
from jax import jit, random
from numpyro import handlers
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid

from benchmarks.disease_mapping import model as numpyro_model
from benchmarks.meta_learning.gp import wandb_2d_plots
from dl4bi.core.train import Callback, evaluate, save_ckpt, train
from dl4bi.meta_learning.data.spatial import SpatialBatch, SpatialData
from dl4bi.meta_learning.data.utils import batch_BLD, permute_L_in_BLD
from dl4bi.meta_learning.utils import cfg_to_run_name


@hydra.main("configs", "training", None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs,
        group="gnp",
    )
    wandb.log_artifact(getsourcefile(numpyro_model), "model.py")
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    train_dataloader = valid_dataloader = build_dataloader(cfg.data)
    # TODO: fix this
    # callbacks = [Callback(wandb_2d_plots, cfg.plot_interval)]
    # callback_dataloader = build_grid_dataloader(cfg.data)
    callbacks, callback_dataloader = [], None
    optimizer = instantiate(cfg.optimizer)
    model = instantiate(cfg.model)
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
        valid_dataloader,
        callbacks=callbacks,
        callback_dataloader=callback_dataloader,
    )
    metrics = evaluate(
        rng_test,
        state,
        model.valid_step,
        valid_dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = f"results/{cfg.project}/{cfg.data.name}/{cfg.seed}/{run_name}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


@partial(jit, static_argnames="B")
def sample_prior(rng, s: jax.Array, n: jax.Array, B: jax.Array):
    trace = handlers.trace(handlers.seed(numpyro_model.survey_model, rng)).get_trace(
        s, n, None, sample_shape=(B,)
    )
    z = trace["z"]["value"]
    n_pos = trace["n_pos"]["value"]
    return z, n_pos


@partial(jit, static_argnames="shape")
def sample_n(rng, shape):
    return jnp.ceil(300 / random.gamma(rng, 3, shape)).astype(jnp.int32)


@partial(
    jit, static_argnames=["num_ctx_min", "num_ctx_max", "num_test", "test_includes_ctx"]
)
def make_batch(
    rng,
    s: jax.Array,
    n: jax.Array,
    x: jax.Array | None,
    z: jax.Array,
    n_pos: jax.Array,
    *,
    num_ctx_min,
    num_ctx_max,
    num_test,
    test_includes_ctx,
):
    """
    Adapted from `meta_learning.data.spatial._batch`,
    but accomodates the idea of observing `n_pos` but predicting `z = logit(theta)`.
    """

    rng_p, rng_b = random.split(rng)
    S_to_L = jit(lambda v: v.reshape(v.shape[0], -1, v.shape[-1]))
    batch_args = (num_ctx_min, num_ctx_max, num_test, test_includes_ctx)

    s_shape = s.shape
    s, n, z, n_pos = n[..., None], z[..., None], n_pos[..., None]
    if x is None:
        s, n, z, n_pos = map(S_to_L, [s, n, z, n_pos])
        s, n, z, n_pos = map(jnp.float32, [s, n, z, n_pos])
        assert s.ndim == n.ndim == z.ndim == n_pos.ndim == 3, "Expected 3D arrays"

        s, n, z, n_pos, inv_permute_idx = permute_L_in_BLD(rng_p, [s, n, z, n_pos])
        s_c, n_c, z_c, n_pos_c, mask_c, s_t, n_t, z_t, n_pos_t, mask_t = batch_BLD(
            rng_b, [s, n, z, n_pos], *batch_args
        )
        x_c, x_t = None, None
    else:
        s, x, n, z, n_pos = map(S_to_L, [s, x, n, z, n_pos])
        s, x, n, z, n_pos = map(jnp.float32, [s, x, n, z, n_pos])
        assert s.ndim == x.ndim == n.ndim == z.ndim == n_pos.ndim == 3, (
            "Expected 3D arrays"
        )
        s, x, n, z, n_pos, inv_permute_idx = permute_L_in_BLD(
            rng_p, [s, x, n, z, n_pos]
        )
        s_c, x_c, n_c, z_c, n_pos_c, mask_c, s_t, x_t, n_t, z_t, n_pos_t, mask_t = (
            batch_BLD(rng_b, [s, x, n, z, n_pos], *batch_args)
        )

    z_c_obs = jsp.special.logit(n_pos_c / n_c)

    # Various options for feeding the context
    # f_c = z_c_obs
    # f_c = jnp.concat([z_c_obs, n_c], axis=-1)
    f_c = jnp.concat([n_pos_c, n_c], axis=-1)
    # f_c = jnp.concat([n_pos_c / n_c, n_c], axis=-1)

    # Target theta
    f_t = jax.nn.sigmoid(z_t)

    return SpatialBatch(
        x_c,
        s_c,
        f_c,
        mask_c,
        x_t,
        s_t,
        f_t,
        mask_t,
        inv_permute_idx,
        s_shape,
    )


def build_dataloader(cfg: DictConfig):
    """
    Generates samples from `model.spatial_effect`.
    """

    B, L, D = cfg.batch_size, cfg.num_test, len(cfg.s)
    s_min = jnp.array([axis["start"] for axis in cfg.s])
    s_max = jnp.array([axis["stop"] for axis in cfg.s])
    batchify = jit(lambda x: jnp.repeat(x[None, ...], B, axis=0))

    def dataloader(rng: jax.Array):
        while True:
            rng_s, rng_n, rng_sp, rng_b, rng = random.split(rng, 5)
            s = random.uniform(rng_s, (L, D), jnp.float32, s_min, s_max)
            n = sample_n(rng_n, (L,))
            z, n_pos = sample_prior(rng_sp, s, n, B)
            s, n = batchify(s), batchify(n)
            yield make_batch(
                rng_b,
                s,
                n,
                None,
                z,
                n_pos,
                num_ctx_min=cfg.num_ctx.min,
                num_ctx_max=cfg.num_ctx.max,
                num_test=cfg.num_test,
                test_includes_ctx=True,
            )

    return dataloader


# def build_grid_dataloader(cfg: DictConfig):
#     # broken
#     # TODO fix
#     B = cfg.batch_size
#     to_extra = lambda d: {k: v.item() for k, v in d.items() if v is not None}

#     s = build_grid(cfg.s)
#     s = jnp.repeat(s[None, ...], B, axis=0)
#     L_test = s.shape[1] * s.shape[2]

#     def dataloader(rng: jax.Array):
#         while True:
#             rng_sp, rng_b, rng = random.split(rng, 3)
#             n = sample_n(rng_n, (L, D))
#             z, n_pos = sample_prior(rng_sp, s[0], n, B)
#             y = y.reshape(*s.shape[:-1], y.shape[-1])
#             d = SpatialData(None, s, y)
#             b = d.batch(
#                 rng_b,
#                 cfg.num_ctx.min,
#                 cfg.num_ctx.max,
#                 num_test=L_test,
#                 test_includes_ctx=True,
#                 obs_noise=None,
#             )
#             yield b, to_extra(dict())

#     return dataloader


if __name__ == "__main__":
    main()
