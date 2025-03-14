#!/usr/bin/env python3
from pathlib import Path
from typing import Callable

import hydra
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import optax
import pandas as pd
import wandb
from hydra.utils import instantiate
from jax import jit, random
from jax.scipy.special import logsumexp
from numpyro.infer import MCMC, NUTS, Predictive, log_likelihood
from omegaconf import DictConfig, OmegaConf

from dl4bi.core.train import evaluate, save_ckpt, train
from dl4bi.meta_learning.data.tabular import TabularBatch, TabularData
from dl4bi.meta_learning.utils import cfg_to_run_name

# TODO(danj): finish this - how can you meta-learn non-iid data?
# need to create (ctx, test) that share the same parameters, but use different counts
# TODO(danj): run inference
# TODO(danj): calculate pointwise log likelihood


@hydra.main("configs/baseball", config_name="default", version_base=None)
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
    dataloader, test_dataloader = build_dataloaders(cfg.data)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.clip_max_norm),
        optax.yogi(1e-3),
    )
    model = instantiate(cfg.model)
    state = train(
        rng_train,
        model,
        optimizer,
        model.train_step,
        model.valid_step,
        dataloader,
        dataloader,
        dataloader,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
    )
    metrics = evaluate(
        rng_test,
        state,
        model.valid_step,
        test_dataloader,
        num_steps=1,
    )
    batch = next(test_dataloader(rng_test))
    output = state.apply_fn({"params": state.params, **state.kwargs}, **batch)
    print(output.mu)
    print(batch.f_test)
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloaders(cfg: DictConfig):
    (train_at_bats, train_hits), (test_at_bats, test_hits) = load_baseball_dataset()
    B, L = cfg.batch_size, train_at_bats.shape[0]
    ids = jnp.repeat(jnp.arange(L)[None, :, None], B, axis=0)  # [B, L, 1]
    prior_pred = jit(Predictive(partially_pooled, num_samples=B))
    post_pred = jit(
        lambda rng, z, x: Predictive(partially_pooled, z, num_samples=B)(rng, x)
    )
    batchify = jit(lambda x: jnp.repeat(x[None, :, None], B, axis=0))

    def train_dataloader(rng):
        while True:
            rng_x, rng_p1, rng_p2, rng = random.split(rng, 4)
            x_ctx, x_test = random.randint(rng_x, (2, L), minval=40, maxval=650)
            ctx_samples = prior_pred(rng_p1, x_ctx)
            f_ctx = ctx_samples.pop("obs")[..., None]  # [B, L, 1]
            test_samples = post_pred(rng_p2, ctx_samples, x_test)
            f_test = test_samples["obs"][..., None]  # [B, L, 1]
            x_ctx, x_test = batchify(x_ctx), batchify(x_test)
            x_ctx = jnp.concat([ids, x_ctx], axis=-1)  # [B, L, 2]
            x_test = jnp.concat([ids, x_test], axis=-1)  # [B, L, 2]
            mask_ctx = None
            yield TabularBatch(x_ctx, f_ctx, mask_ctx, x_test, f_test)

    def test_dataloader(rng):
        ids = jnp.arange(L)[None, :, None]
        x_ctx = train_at_bats[None, :, None]
        x_test = test_at_bats[None, :, None]
        x_ctx = jnp.concat([ids, x_ctx], axis=-1)  # [B=1, L, 2]
        x_test = jnp.concat([ids, x_test], axis=-1)
        mask_ctx = jnp.ones((1, L), dtype=bool)
        f_ctx = train_hits[None, :, None]  # [B=1, L, 1]
        f_test = test_hits[None, :, None]
        yield TabularBatch(x_ctx, f_ctx, mask_ctx, x_test, f_test)

    return train_dataloader, test_dataloader


# source: https://num.pyro.ai/en/stable/examples/baseball.html
def partially_pooled(at_bats, hits=None):
    r"""
    Number of hits has a Binomial distribution with independent
    probability of success, $\phi_i$. Each $\phi_i$ follows a Beta
    distribution with concentration parameters $c_1$ and $c_2$, where
    $c_1 = m * kappa$, $c_2 = (1 - m) * kappa$, $m ~ Uniform(0, 1)$,
    and $kappa ~ Pareto(1, 1.5)$.

    :param (jnp.ndarray) at_bats: Number of at bats for each player.
    :param (jnp.ndarray) hits: Number of hits for the given at bats.
    :return: Number of hits predicted by the model.
    """
    m = numpyro.sample("m", dist.Uniform(0, 1))
    kappa = numpyro.sample("kappa", dist.Pareto(1, 1.5))
    num_players = at_bats.shape[0]
    with numpyro.plate("num_players", num_players):
        phi_prior = dist.Beta(m * kappa, (1 - m) * kappa)
        phi = numpyro.sample("phi", phi_prior)
        return numpyro.sample("obs", dist.Binomial(at_bats, probs=phi), obs=hits)


def load_baseball_dataset():
    url = "https://raw.githubusercontent.com/pyro-ppl/datasets/refs/heads/master/EfronMorrisBB.txt"
    path = Path("cache/baseball.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(url, sep="\t")
        df.to_csv("cache/baseball.csv", index=False)
    at_bats, hits = df["At-Bats"].values, df["Hits"].values
    season_at_bats, season_hits = df["SeasonAt-Bats"].values, df["SeasonHits"].values
    return (at_bats, hits), (season_at_bats, season_hits)


def run_inference(
    rng: jax.Array,
    model: Callable,
    at_bats: jax.Array,
    hits: jax.Array,
    cfg: DictConfig,
):
    mcmc = MCMC(
        NUTS(model),
        num_warmup=cfg.num_warmup,
        num_samples=cfg.num_samples,
        num_chains=cfg.num_chains,
        progress_bar=True,
    )
    mcmc.run(rng, at_bats, hits)
    return mcmc.get_samples()


if __name__ == "__main__":
    main()
