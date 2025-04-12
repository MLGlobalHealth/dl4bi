import pickle
from pathlib import Path

import arviz as az
import hydra
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS, init_to_median
from omegaconf import DictConfig

from benchmarks.disease_mapping.data import get_shape, get_survey_data
from benchmarks.disease_mapping.model import survey_model
from benchmarks.disease_mapping.visualize import plot_surveys


def run_mcmc(cfg: DictConfig, data: dict[str, jax.Array]) -> MCMC:
    sampler = NUTS(survey_model, init_strategy=init_to_median())
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


@hydra.main("configs", "inference", None)
def main(cfg: DictConfig):
    """
    Run MCMC inference on the survey model.
    """
    results_path = Path("results")
    results_path.mkdir(parents=True, exist_ok=True)

    # Load data
    data = get_survey_data(**cfg.data)

    # Run MCMC
    mcmc = run_mcmc(cfg.mcmc, data)

    # Save results
    with open(results_path / "mcmc.pickle", "wb") as f:
        pickle.dump(mcmc, f)
    # Also save the data used
    jnp.savez(results_path / "data.npz", **data)
    shape = get_shape(cfg.data.iso, cfg.data.get("region"))
    fig = plot_surveys(**data, shape=shape)
    fig.savefig(results_path / "data.png", dpi=300)

    # Show summary
    run = az.from_numpyro(mcmc)
    print(az.summary(run, hdi_prob=0.95))


if __name__ == "__main__":
    main()
