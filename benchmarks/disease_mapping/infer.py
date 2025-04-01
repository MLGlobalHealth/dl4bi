import arviz as az
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_median

from benchmarks.disease_mapping.data import prepare_data
from benchmarks.disease_mapping.model import model, predict_gp, predict_np

numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)
numpyro.enable_x64()


def main():
    s, Np, N = prepare_data()

    mask = Np / N > 0.5
    s, Np, N = s[mask], Np[mask], N[mask]

    # example data
    # s = jnp.array([[0, 0], [1, 1]])
    # Np = jnp.array([0, 0])
    # N = jnp.array([2, 3])

    print(jnp.vstack([s.T, Np, N, Np / N]))

    rng = jax.random.key(42)

    sampler = NUTS(model, init_strategy=init_to_median())
    mcmc = MCMC(
        sampler,
        num_warmup=0,
        num_samples=1000,
        num_chains=1,
    )

    mcmc.run(rng, s, Np, N)

    data = az.from_numpyro(mcmc)
    print(az.summary(data))

    # posterior = az.from_numpyro(mcmc)
    # print(az.summary(posterior))

    # prior_pred = Predictive(model, num_samples=100)
    # prior_pred()
    # print(prior_pred)


# from numpyro import handlers


# def log_likelihood(model, *args, **kwargs):
#     rng = jax.random.key(int(time()))
#     model = handlers.seed(model, rng)
#     model_trace = handlers.trace(model).get_trace(*args, **kwargs)
#     obs_node = model_trace["Np"]

#     return obs_node["fn"].log_prob(obs_node["value"])


# log_likelihood(model, s, Np, N)


if __name__ == "__main__":
    main()
