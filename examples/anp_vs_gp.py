#!/usr/bin/env python3
import argparse
import pickle
import sys
from dataclasses import dataclass

import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from jax.tree_util import Partial
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from sps.gp import GP
from sps.kernels import Kernel, matern_3_2, periodic
from sps.priors import Prior
from sps.utils import build_grid


@dataclass
class Task:
    name: str
    kernel: Kernel
    var: Prior
    ls: Prior


def main(num_warmup, num_samples):
    key = random.key(42)
    num_points, num_train, obs_noise, period = 128, 32, 0.05, 0.2
    s = build_grid([{"start": 0, "stop": 1, "num": num_points}]).squeeze()
    rng_noise, rng_idx, rng_gp, rng_hmc = random.split(key, 4)
    obs_noise = obs_noise * random.normal(rng_noise, (num_points,))
    idx = random.choice(rng_idx, len(s), (num_train,), replace=False)
    periodic_0_1 = Partial(periodic, period=period)
    var, ls = Prior("fixed", {"value": 1.0}), Prior("fixed", {"value": 0.2})
    periodic_task = Task(name="Periodic", kernel=periodic_0_1, var=var, ls=ls)
    matern_3_2_task = Task(name="Matern 3-2", kernel=matern_3_2, var=var, ls=ls)
    for task in [periodic_task, matern_3_2_task]:
        _, _, _, f = GP(task.kernel, task.var, task.ls).simulate(rng_gp, s, 1)
        f = f.squeeze()
        f_noisy = f + obs_noise
        m = build_gp_model(task.kernel)
        pp = hmc(rng_hmc, m, s, f_noisy, idx, num_warmup, num_samples)
        pp.update({"s": s, "f": f, "f_noisy": f_noisy, "idx": idx})
        pp_name = task.name.lower().replace(" ", "_")
        with open(f"{pp_name}_gp_post_pred.pkl", "wb") as of:
            pickle.dump(pp, of)
        plot_posterior_predictive_samples(task.name, s, f, f_noisy, idx, pp["obs"])


def build_gp_model(kernel):
    def m(s, f, idx):
        variance = numpyro.sample("variance", dist.HalfNormal())
        lengthscale = numpyro.sample("lengthscale", dist.HalfNormal())
        # jitter added on diagonal for cholesky decomposition stability
        K = kernel(s, s, variance, lengthscale) + 1e-6 * jnp.eye(len(s))
        f_mu = numpyro.sample("f_mu", dist.MultivariateNormal(0, K))
        f_sigma = numpyro.sample("f_sigma", dist.HalfNormal(0.1))
        obs, f_mu = (f, f_mu) if f is None else (f[idx], f_mu[idx])
        numpyro.sample("obs", dist.Normal(f_mu, f_sigma), obs=obs)

    return m


def hmc(rng_hmc, model, s, f, idx, num_warmup=100, num_samples=200):
    rng_mcmc, rng_pp = random.split(rng_hmc)
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_mcmc, s, f, idx)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    return Predictive(model, samples)(rng_pp, s, None, idx)


def dataloader(key, gp, locations, batch_size=64, approx=False):
    while True:
        rng, key = random.split(key)
        yield gp.simulate(rng, locations, batch_size, approx)


def plot_posterior_predictive_samples(
    model_name,
    s,
    f,
    f_noisy,
    idx,
    pp_samples,
    hdi_prob=0.9,
):
    f_hat = np.array(pp_samples)
    f_hat_mu = f_hat.mean(axis=0)
    f_hat_hdi = az.hdi(f_hat)
    plt.plot(s, f, color="black")
    plt.plot(s, f_hat_mu, color="steelblue")
    plt.scatter(s[idx], f_noisy[idx], color="black")
    plt.fill_between(
        s,
        f_hat_hdi[:, 0],
        f_hat_hdi[:, 1],
        alpha=0.4,
        color="steelblue",
        interpolate=True,
    )
    ax = plt.gca()
    ax.set_xlabel("s")
    ax.set_ylabel("f")
    plt.title(f"{model_name} Posterior Predictive")
    plt.savefig(f"{model_name} Posterior Predictive.pdf", dpi=600)
    plt.clf()


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-nw", "--num_warmup", default=500, type=int)
    parser.add_argument("-ns", "--num_samples", default=1000, type=int)
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args.num_warmup, args.num_samples)
