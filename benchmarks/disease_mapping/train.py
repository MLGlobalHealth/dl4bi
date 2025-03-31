from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import wandb
from hydra.utils import instantiate
from jax import jit, random
from numpyro import handlers
from omegaconf import DictConfig, OmegaConf

from benchmarks.disease_mapping.model import spatial_process
from dl4bi.core.train import evaluate, save_ckpt, train
from dl4bi.meta_learning.data.spatial import SpatialData
from dl4bi.meta_learning.utils import cfg_to_run_name


@hydra.main("configs", "default", None)
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
    # TODO @pgrynfelder: implement this
    # clbk
    # clbk_dataloader = build_dataloader(cfg.data, is_callback=True)
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
        # callbacks=[Callback(clbk, cfg.plot_interval)],
        # callback_dataloader=clbk_dataloader,
    )
    metrics = evaluate(
        rng_test,
        state,
        model.valid_step,
        valid_dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    kernel = cfg.kernel._target_.split(".")[-1]
    path = f"results/{cfg.project}/{cfg.data.name}/{kernel}/{cfg.seed}/{run_name}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloader(data: DictConfig):
    """
    Generates samples from `model.spatial_process`.
    """

    B, L, D = data.batch_size, data.num_test, len(data.s)
    s_min = jnp.array([axis["start"] for axis in data.s])
    s_max = jnp.array([axis["stop"] for axis in data.s])
    batchify = jit(lambda x: jnp.repeat(x[None, ...], B, axis=0))

    @partial(jit, static_argnames=["B"])
    def sp(rng: jax.Array, s: jax.Array, B: int):
        f = handlers.seed(spatial_process, rng)
        return f(s, (B,))

    def dataloader(rng: jax.Array):
        while True:
            rng_s, rng_sp, rng_b, rng = random.split(rng, 4)
            s = random.uniform(rng_s, (L, D), jnp.float32, s_min, s_max)
            y = sp(rng_sp, s, B)
            s = batchify(s)
            d = SpatialData(None, s, y)
            b = d.batch(
                rng_b,
                data.num_ctx.min,
                data.num_ctx.max,
                num_test=L,
                test_includes_ctx=False,
                obs_noise=None,
            )
            yield b

    return dataloader


if __name__ == "__main__":
    main()
