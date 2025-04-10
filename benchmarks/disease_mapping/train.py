from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import wandb
from hydra.utils import instantiate
from jax import jit, random
from numpyro import handlers
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid

from benchmarks.disease_mapping.model import spatial_effect
from benchmarks.meta_learning.gp import wandb_2d_plots
from dl4bi.core.train import Callback, evaluate, save_ckpt, train
from dl4bi.meta_learning.data.spatial import SpatialData
from dl4bi.meta_learning.utils import cfg_to_run_name


@hydra.main("configs", "training", None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    train_dataloader = valid_dataloader = build_dataloader(cfg.data)
    # TODO: fix this for spatiotemportal data
    clbk = wandb_2d_plots
    clbk_dataloader = build_grid_dataloader(cfg.data)
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
        callbacks=[Callback(clbk, cfg.plot_interval)],
        callback_dataloader=clbk_dataloader,
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


def build_dataloader(cfg: DictConfig):
    """
    Generates samples from `model.spatial_effect`.
    """

    B, L, D = cfg.batch_size, cfg.num_test + cfg.num_ctx.max, len(cfg.s)
    s_min = jnp.array([axis["start"] for axis in cfg.s])
    s_max = jnp.array([axis["stop"] for axis in cfg.s])
    batchify = jit(lambda x: jnp.repeat(x[None, ...], B, axis=0))

    @jit
    def sp(rng: jax.Array, s: jax.Array):
        f = handlers.seed(spatial_effect, rng)
        return f(s, (B,))[..., None]

    def dataloader(rng: jax.Array):
        while True:
            rng_s, rng_sp, rng_b, rng = random.split(rng, 4)
            s = random.uniform(rng_s, (L, D), jnp.float32, s_min, s_max)
            y = sp(rng_sp, s)
            s = batchify(s)
            d = SpatialData(None, s, y)
            b = d.batch(
                rng_b,
                cfg.num_ctx.min,
                cfg.num_ctx.max,
                num_test=cfg.num_test,
                test_includes_ctx=False,
                obs_noise=None,
            )
            yield b

    return dataloader


def build_grid_dataloader(cfg: DictConfig):
    B = cfg.batch_size
    to_extra = lambda d: {k: v.item() for k, v in d.items() if v is not None}

    s = build_grid(cfg.s)
    s = jnp.repeat(s[None, ...], B, axis=0)
    L_test = s.shape[1] * s.shape[2]

    @jit
    def sp(rng: jax.Array):
        f = handlers.seed(spatial_effect, rng)
        # TODO: return kernel params
        y = f(s[0], (B,))[..., None]
        return y

    def dataloader(rng: jax.Array):
        while True:
            rng_sp, rng_b, rng = random.split(rng, 3)
            y = sp(rng_sp)
            y = y.reshape(*s.shape[:-1], y.shape[-1])
            d = SpatialData(None, s, y)
            b = d.batch(
                rng_b,
                cfg.num_ctx.min,
                cfg.num_ctx.max,
                num_test=L_test,
                test_includes_ctx=True,
                obs_noise=None,
            )
            yield b, to_extra(dict())

    return dataloader


if __name__ == "__main__":
    main()
