import pickle
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from benchmarks.disease_mapping.data import get_grid, get_shape
from benchmarks.disease_mapping.model import (
    get_np_sampler,
    sample_gp,
    sample_prevalence,
)
from benchmarks.disease_mapping.utils import batch, unbatch
from benchmarks.disease_mapping.visualize import plot_predictions
from dl4bi.core.train import load_ckpt
from dl4bi.core.utils import breakpoint_if_nonfinite


def predict(s_c, samples, seed, model, batch_size, iso, region, res):
    """
    Transforms samples $y_c (+params) | (s, np, n)_c$ into $y_t | s_t, (s, np, n)_c$.
    """
    rng = jax.random.key(seed)
    model = model
    batch_size = batch_size
    s_t = get_grid(iso, region, res)

    print("Num samples:", samples["y"].shape[0])
    print("Num context locations:", s_c.shape[0])
    print("Num target locations:", s_t.shape[0])

    s_c = jnp.repeat(s_c[None], batch_size, axis=0)
    s_t = jnp.repeat(s_t[None], batch_size, axis=0)
    samples = {k: batch(v, batch_size) for k, v in samples.items()}
    y_c = samples.pop("y")
    num_iters = y_c.shape[0]

    if model.lower() == "gp":
        print("Using GP for predictions.")
        sample_conditioned_sp = sample_gp
        model_name = "gp"
    else:
        state, cfg_model = load_ckpt(model)
        print(f"Using {cfg_model.name} for predictions.")
        sample_conditioned_sp = get_np_sampler(state)
        model_name = cfg_model.name

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

    breakpoint_if_nonfinite(theta_t)

    return model_name, s_t[0], unbatch(theta_t)


@hydra.main("configs", "inference", None)
def main(cfg: DictConfig):
    mcmc_results_path = Path("results")

    # Load data
    data = dict(jnp.load(mcmc_results_path / "data.npz"))
    with open(mcmc_results_path / "mcmc.pickle", "rb") as f:
        mcmc = pickle.load(f)
    samples = mcmc.get_samples()

    # Predict
    model_name, s_t, theta_t = predict(data["s"], samples, **cfg.prediction)

    # Save results
    results_path = Path("results") / model_name
    results_path.mkdir(parents=True, exist_ok=True)
    jnp.savez(results_path / "predictions.npz", s=s_t, theta=theta_t)

    # Plot results
    fig = plot_predictions(
        s_t,
        theta_t,
        get_shape(cfg.prediction.iso, cfg.prediction.get("region")),
        data,
    )
    fig.savefig(
        results_path / "predictions.png",
        dpi=300,
    )


if __name__ == "__main__":
    main()
