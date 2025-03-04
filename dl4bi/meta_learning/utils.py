import re
from pathlib import Path
from typing import Union

import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf

from ..core.train import load_ckpt


def cfg_to_run_name(cfg: DictConfig):
    name = cfg.model.cls
    if "TNPKR" in name:
        prefix = "model.kwargs.blk.kwargs.attn."
        attn_cls = OmegaConf.select(cfg, prefix + "cls")
        if attn_cls == "MultiHeadAttention":
            attn_cls = OmegaConf.select(cfg, prefix + "kwargs.attn.cls")
        name += ": " + attn_cls
    return name


def load_ckpts(
    dir: Union[str, Path],
    only_regex: Union[str, re.Pattern] = r".*",
    exclude_regex: Union[str, re.Pattern] = "$^",
):
    """Loads all checkpoints in a given base dir."""
    ckpt = {}
    if isinstance(only_regex, str):
        only_regex = re.compile(only_regex, re.IGNORECASE)
    if isinstance(exclude_regex, str):
        exclude_regex = re.compile(exclude_regex, re.IGNORECASE)
    for p in Path(dir).glob("*.ckpt"):
        if only_regex.match(str(p)) and not exclude_regex.match(str(p)):
            state, tmp_cfg = load_ckpt(p)
            ckpt[cfg_to_run_name(tmp_cfg)] = {"state": state, "cfg": tmp_cfg}
    return ckpt


def regression_to_rgb(f: jax.Array):
    return jnp.clip(f / 2 + 0.5, 0, 1)  # [-1, 1] => [0, 1]
