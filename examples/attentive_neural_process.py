#!/usr/bin/env python
import hydra
import jax.numpy as jnp
from jax import jit, vmap
from omegaconf import DictConfig, OmegaConf

from dge import *


@hydra.main("configs", "attentive_neural_process", None)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    task = cfg.task
    func = get_func(task.name, task.kwargs.frequency)
    model = instantiate(OmegaConf.to_container(cfg.model, resolve=True))
    print(model)


def get_func(name: str, freq: float):
    match name:
        case "sine":
            return vmap(lambda s: jnp.sin(2 * jnp.pi * freq * s))
        case "cosine":
            return vmap(lambda s: jnp.cos(2 * jnp.pi * freq * s))
    raise ValueError(f"Invalid function: {name}")


def instantiate(d: dict):
    for k in d:
        if isinstance(d[k], dict):
            d[k] = instantiate(d[k])
    if "cls" in d:
        cls, kwargs = d["cls"], d.get("kwargs", {})
        return globals()[cls](**kwargs)
    return d


if __name__ == "__main__":
    main()
