from typing import Union

import flax.linen as nn
import jax.numpy as jnp
from inference_models import *  # noqa: F403
from models import *  # noqa: F403
from numpyro.distributions import *  # noqa: F403
from omegaconf import DictConfig, OmegaConf
from sps.kernels import matern_1_2, matern_3_2, matern_5_2, periodic, rbf

from dl4bi.vae.train_utils import TrainState


def generate_model_name(cfg: DictConfig):
    spatial_prior = cfg.inference_model.spatial_prior.func
    dec_name = cfg.model.kwargs.decoder.cls
    return cfg.get("name", f"{cfg.model.cls}_{dec_name}_{spatial_prior}")


def instantiate(d: Union[dict, DictConfig]):
    """Convenience function to instantiate objects config."""
    if isinstance(d, DictConfig):
        d = OmegaConf.to_container(d, resolve=True)
    if "cls" in d:
        cls, kwargs = d["cls"], d.get("kwargs", {})
        for k in kwargs:
            if k == "act_fn":
                kwargs[k] = getattr(nn, kwargs[k])
            elif isinstance(kwargs[k], dict):
                kwargs[k] = instantiate(kwargs[k])
        return globals()[cls](**kwargs)
    if "numpyro_dist" in d:  # Case for NumPyro distributions
        dist_cls, kwargs = d["numpyro_dist"], d.get("kwargs", {})
        kwargs = {k: jnp.array(i) for k, i in kwargs.items()}
        return globals()[dist_cls](**kwargs)
    elif "func" in d:
        return eval(d["func"])
    return d


# NOTE: placeholder prior functions, to allow similar initialization across all spatial priors
def car():
    pass


def iid():
    pass
