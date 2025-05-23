import pickle
from cProfile import label
from datetime import datetime
from inspect import getsource
from pathlib import Path
from timeit import default_timer as timer

import arviz as az
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from omegaconf import DictConfig, OmegaConf

from benchmarks.disease_mapping import survey_model
from benchmarks.disease_mapping.data import get_survey_data
from benchmarks.disease_mapping.visualize import plot_surveys


def run_mcmc(cfg: DictConfig, data: dict[str, jax.Array]) -> MCMC:
    sampler = NUTS(survey_model.model)
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


def prior_predictive_check(data: dict[str, jax.Array]):
    rng = jax.random.key(0)
    s, n, n_pos, x = data["s"], data["n"], data["n_pos"], data.get("x", None)
    n = data["n"].astype(jnp.int32)
    L = n.shape[-1]

    predictive = Predictive(survey_model.model, num_samples=100)
    samples = predictive(rng, s, n, None, x)
    prior_n_pos = samples["n_pos"]

    ci = 95
    lo = jnp.percentile(prior_n_pos, (100 - ci) / 2, axis=0)
    hi = jnp.percentile(prior_n_pos, 100 - (100 - ci) / 2, axis=0)
    y = jnp.stack([lo, hi], axis=-1)
    x = jnp.arange(L)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.fill_between(
        x, lo, hi, step="mid", alpha=0.3, label=f"Prior predictive {ci}% CI"
    )
    ax.scatter(x, n_pos, s=1, color="red", label="Observed data")
    ax.legend()
    return fig


def cfg_to_name(cfg: DictConfig) -> str:
    if cfg.get("name") is not None:
        return "MCMC_" + cfg.name
    else:

        def shorten_key(key: str):
            return "".join(x[0] for x in key.split("_"))

        values_repr = [
            shorten_key(k) + ":" + str(v)
            # for k, v in sorted(d.items(), key=lambda x: x[0])
            for k, v in cfg.items()
            if v is not None
        ]

        return "MCMC_" + "_".join(values_repr)


@hydra.main("configs", "mcmc", None)
def main(cfg: DictConfig):
    """
    Run MCMC inference on the survey model.
    """
    print("Running MCMC with config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    name = cfg_to_name(cfg)
    results_path = Path("results") / name
    results_path.mkdir(parents=True, exist_ok=True)

    # Load data
    data = get_survey_data(
        cfg.iso,
        cfg.region,
        cfg.query,
        cfg.urban_rural,
        cfg.time,
    )

    # Dump config, data, and model
    OmegaConf.save(cfg, results_path / "config.yaml", resolve=True)
    jnp.savez(results_path / "data.npz", **data)
    fig = plot_surveys(data)
    fig.savefig(results_path / "data.png", dpi=300)
    (results_path / "model.txt").write_text(getsource(survey_model))

    # Prior predictive check
    fig = prior_predictive_check(data)
    fig.savefig(results_path / "prior_predictive.png", dpi=300)

    # Run MCMC
    timer_start = timer()
    mcmc = run_mcmc(cfg, data)
    timer_end = timer()
    print(f"MCMC took {timer_end - timer_start:.2f} seconds.")
    (results_path / "time.txt").write_text(f"{timer_end - timer_start:.2f} seconds")
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
