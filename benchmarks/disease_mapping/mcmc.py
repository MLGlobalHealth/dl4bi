import pickle
from inspect import getsource
from pathlib import Path

import arviz as az
import hydra
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS, init_to_median
from omegaconf import DictConfig, OmegaConf

from benchmarks.disease_mapping.data import get_shape, get_survey_data
import benchmarks.disease_mapping.model as model
from benchmarks.disease_mapping.visualize import plot_surveys

from datetime import datetime


def run_mcmc(cfg: DictConfig, data: dict[str, jax.Array]) -> MCMC:
    sampler = NUTS(model.survey_model, init_strategy=init_to_median())
    mcmc = MCMC(
        sampler,
        num_warmup=cfg.num_warmup,
        num_samples=cfg.num_samples,
        num_chains=cfg.num_chains,
        chain_method=jax.vmap if cfg.chain_method == "vmap" else cfg.chain_method,
        progress_bar=cfg.progress_bar,
    )
    rng = jax.random.key(cfg.seed)
    mcmc.run(rng, **data)

    return mcmc


@hydra.main("configs", "mcmc", None)
def main(cfg: DictConfig):
    """
    Run MCMC inference on the survey model.
    """
    results_path = Path("results") / (
        "MCMC_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    results_path.mkdir(parents=True, exist_ok=True)

    # Load data
    data = get_survey_data(
        cfg.iso,
        cfg.region,
        cfg.query,
        cfg.res,
    )
    # Dump config, data, and model
    OmegaConf.save(cfg, results_path / "config.yaml", resolve=True)
    jnp.savez(results_path / "data.npz", **data)
    shape = get_shape(cfg.iso, cfg.region) if cfg.iso else None
    fig = plot_surveys(**data, shape=shape)
    fig.savefig(results_path / "data.png", dpi=300)
    (results_path / "model.txt").write_text(getsource(model.survey_model))

    # Run MCMC
    mcmc = run_mcmc(cfg, data)

    # Save results
    with open(results_path / "mcmc.pickle", "wb") as f:
        pickle.dump(mcmc, f)

    # Summary
    run = az.from_numpyro(mcmc)
    summary = az.summary(run, hdi_prob=0.95)
    summary.to_csv(results_path / "summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
