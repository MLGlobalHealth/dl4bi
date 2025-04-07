import pickle
from os import environ
from pathlib import Path

import arviz as az
import hydra
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS, init_to_median
from omegaconf import DictConfig

from benchmarks.disease_mapping.data import get_grid, get_survey_data
from benchmarks.disease_mapping.model import (
    get_np_sampler,
    sample_gp,
    sample_prevalence,
    survey_model,
)
from benchmarks.disease_mapping.utils import batch, unbatch
from benchmarks.disease_mapping.visualize import plot
from dl4bi.core.train import load_ckpt


def run_mcmc(cfg: DictConfig, data: tuple[jax.Array, ...]) -> MCMC:
    sampler = NUTS(survey_model, init_strategy=init_to_median())
    mcmc = MCMC(
        sampler,
        num_warmup=cfg.num_warmup,
        num_samples=cfg.num_samples,
        num_chains=cfg.num_chains,
        chain_method=jax.vmap if cfg.chain_method == "vmap" else cfg.chain_method,
        progress_bar=cfg.progress_bar,
        thinning=cfg.thinning,
        # jit_model_args=True,
    )
    rng = jax.random.key(cfg.seed)
    mcmc.run(rng, *data)

    return mcmc


def predict(cfg: DictConfig, s_c, samples):
    """
    Transforms samples $y_c (+params) | (s, np, n)_c$ into $y_t | s_t, (s, np, n)_c$.
    """
    rng = jax.random.key(cfg.seed)
    model = cfg.model
    batch_size = cfg.batch_size
    s_t = get_grid(cfg.iso, cfg.region)

    s_c = jnp.repeat(s_c[None], batch_size, axis=0)
    s_t = jnp.repeat(s_t[None], batch_size, axis=0)
    samples = {k: batch(v, batch_size) for k, v in samples.items()}
    y_c = samples.pop("y")
    num_iters = y_c.shape[0]

    if model.lower() == "gp":
        print("Using GP for predictions.")
        sample_conditioned_sp = sample_gp
    else:
        state, cfg_model = load_ckpt(model)
        print(f"Using {cfg_model.name} for predictions.")
        sample_conditioned_sp = get_np_sampler(state)

    def predict_batch(args):
        rng, y_c, params = args
        # Sample (y_t = f(s_t) + noise) | s_t, (s, np, n)_c.
        rng_y_t, rng_theta_t = jax.random.split(rng)
        y_t = sample_conditioned_sp(rng_y_t, s_c, y_c, s_t, **params)

        # Sample theta_t | s_t, (s, np, n)_c
        logit_theta_t = sample_prevalence(rng_theta_t, y_t, **params)

        return jax.nn.sigmoid(logit_theta_t)

    theta_t = jax.lax.map(
        predict_batch,
        (jax.random.split(rng, num_iters), y_c, samples),
    )
    return s_t[0], unbatch(theta_t)


@hydra.main("configs", "inference", None)
def main(cfg: DictConfig):
    cache_dir = Path(cfg.get("cache_dir", "tmp")).resolve()
    environ["CACHE_DIR"] = str(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    s, n_pos, n = get_survey_data(**cfg.data)

    mcmc_file = cache_dir / "mcmc.pickle"
    if False and mcmc_file.exists():
        mcmc = pickle.loads(mcmc_file.read_bytes())
    else:
        mcmc = run_mcmc(cfg.mcmc, (s, n_pos, n))
        with open(mcmc_file, "wb") as f:
            pickle.dump(mcmc, f)

    inference_data = az.from_numpyro(mcmc)
    print(az.summary(inference_data))

    samples = mcmc.get_samples()
    s_t, theta_t_samples = predict(cfg.predict, s, samples)

    jnp.savez("results.npz", theta_samples=theta_t_samples, s=s_t)

    summary = jnp.stack(
        [theta_t_samples.mean(axis=0), theta_t_samples.std(axis=0)], axis=1
    )
    print(theta_t_samples.shape)
    print(summary)

    plot(s_t, theta_t_samples)


if __name__ == "__main__":
    main()
