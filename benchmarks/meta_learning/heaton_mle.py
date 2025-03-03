#!/usr/bin/env python3

import hydra
import jax.numpy as jnp
import pandas as pd
from jax import random
from omegaconf import DictConfig
from sps.kernels import matern_1_2

from dl4bi.core import gp_mle_sgd


@hydra.main("configs/heaton", config_name="default", version_base=None)
def main(cfg: DictConfig):
    num_restarts, num_samples = 10, 4096
    rng = random.key(cfg.seed)
    df = pd.read_csv(cfg.test.path)
    df.Lat -= df.Lat.mean()
    df.Lon -= df.Lon.mean()
    mean, std = df.MaskTemp.mean(), df.MaskTemp.std()
    df["MaskTemp"] = (df.MaskTemp - mean) / std
    obs_idx = df.MaskTemp.notna().values
    obs = df[obs_idx][["Lon", "Lat", "MaskTemp"]].values
    s_obs, f_obs = obs[:, :-1], obs[:, [-1]]
    thetas = []
    for _ in range(num_restarts):
        rng_idx, rng = random.split(rng)
        idx = random.choice(rng_idx, num_samples, (num_samples,), replace=False)
        var, ls, noise = gp_mle_sgd(s_obs[idx], f_obs[idx], matern_1_2)
        print(f"var {var:0.4f}, ls {ls:0.4f}, noise: {noise: 0.4f}")
        thetas += [[var, ls, noise]]
    var_mu, ls_mu, noise_mu = jnp.array(thetas).mean(axis=0)
    print(f"Average: var {var_mu:0.4f}, ls {ls_mu:0.4f}, noise: {noise_mu:0.4f}")


if __name__ == "__main__":
    main()
