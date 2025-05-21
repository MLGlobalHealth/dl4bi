"""
Facilitates training of the disease mapping model.
Learns the map
(s, N, N+)_ctx, s_test -> distribution over logit(theta)_test
"""

import importlib
from inspect import getsourcefile
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import wandb
from hydra.utils import instantiate
from jax import jit, random
from numpyro.infer import Predictive
from omegaconf import DictConfig, OmegaConf

from dl4bi.core.train import evaluate, save_ckpt, train
from dl4bi.meta_learning.data.spatial import SpatialBatch
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
    numpyro_model = importlib.import_module(cfg.numpyro)
    wandb.log_artifact(getsourcefile(numpyro_model))
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    train_dataloader = valid_dataloader = build_dataloader(cfg)
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
    input_format,
    output_format,
):
    """
    Adapted from `meta_learning.data.spatial._batch`,
    but accomodates the idea of observing `n_pos` but predicting `theta`.
    """

    rng_p, rng_b = random.split(rng)
    S_to_L = jit(lambda v: v.reshape(v.shape[0], -1, v.shape[-1]))
    batch_args = (num_ctx_min, num_ctx_max, num_test, test_includes_ctx)

    s_shape = s.shape
    n, z, n_pos = n[..., None], z[..., None], n_pos[..., None]
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

    # Various options for feeding the context
    match input_format:
        case "survey":
            f_c = jnp.concat([n_pos_c, n_c], axis=-1)
        case "theta":
            # empirical theta
            f_c = n_pos_c / n_c
        case "z":
            # emprirical z
            f_c = jsp.special.logit(n_pos_c / n_c)
        case "z_n":
            # emprirical z + survey size
            f_c = jnp.concat([jsp.special.logit(n_pos_c / n_c), z_c], axis=-1)
        case "theta_n":
            # emprirical theta + survey size
            f_c = jnp.concat([n_pos_c / n_c, z_c], axis=-1)

    match output_format:
        case "theta":
            f_t = jax.nn.sigmoid(z_t)
        case "z":
            f_t = z_t

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
    Generates samples from the provided prior numpyro model.
    """

    B, L, D = cfg.data.batch_size, cfg.data.num_test, len(cfg.s)
    s_min = jnp.array([axis["start"] for axis in cfg.s])
    s_max = jnp.array([axis["stop"] for axis in cfg.s])
    has_x = cfg.data.urban_rural
    numpyro_model = importlib.import_module(cfg.numpyro)

    @jit
    def sample_prior(rng, s, n, x=None):
        prior = Predictive(
            numpyro_model.model,
            posterior_samples=None,
            num_samples=1,
            batch_ndims=1,
            return_sites=["z", "n_pos"],
        )
        samples = prior(rng, s, n, None, x)
        return samples["z"], samples["n_pos"]

    def sample_n(rng, sample_shape):
        return random.geometric(rng, 0.02, sample_shape)

    def sample_x(rng, sample_shape):
        # TODO: might need to sample this per context/target, but that messes with the batch generation
        # about 30% of surveys are urban but <1% of the country grid is
        # p = 0.2
        p = 0.5  # does this even matter or can it learn either way?
        urban = random.bernoulli(rng, p, sample_shape).astype(jnp.float32)
        rural = 1 - urban
        return jnp.stack([urban, rural], axis=-1)

    @jit
    def sample_batch(rng):
        rng_s, rng_n, rng_x, rng_sp, rng_b = random.split(rng, 5)
        s = random.uniform(rng_s, (B, L, D), jnp.float32, s_min, s_max)
        n = sample_n(rng_n, (B, L))
        if has_x:
            x = sample_x(rng_x, (B, L))
        else:
            x = None
        z, n_pos = sample_prior(rng, s, n, x)
        return make_batch(
            rng_b,
            s,
            n,
            x,
            z,
            n_pos,
            num_ctx_min=cfg.data.num_ctx.min,
            num_ctx_max=cfg.data.num_ctx.max,
            num_test=cfg.data.num_test,
            test_includes_ctx=True,
            input_format=cfg.input_format,
            output_format=cfg.output_format,
        )

    def dataloader(rng: jax.Array):
        while True:
            rng, rng_i = random.split(rng)
            yield sample_batch(rng_i)

    return dataloader


if __name__ == "__main__":
    main()
