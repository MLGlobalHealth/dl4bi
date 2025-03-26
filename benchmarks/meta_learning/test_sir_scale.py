#!/usr/bin/env python3
import json
from collections import defaultdict
from pathlib import Path
from time import time

import hydra
from hydra.utils import instantiate
from jax import random
from omegaconf import DictConfig
from tqdm import tqdm

from dl4bi.core.train import TrainState, load_ckpt

from .sir import build_spatial_dataloader


@hydra.main("configs/sir", config_name="default", version_base=None)
def main(cfg: DictConfig):
    results_path = Path(f"results/{cfg.project}/{cfg.seed}")
    rng_data, rng_valid = random.split(random.key(cfg.seed))
    dataloader = build_spatial_dataloader(cfg.data, cfg.sim)
    for path in results_path.rglob("*.ckpt"):
        print("=" * 20, path.stem, "=" * 20)
        rng = rng_valid
        state, m_cfg = load_ckpt(path)
        out_fn = path.parent / (path.stem + "_" + cfg.data.name)
        # when these fail and can't be caught, the exception sometimes can't be caught
        if cfg.data.name in ["256x256", "1024x1024"] and path.stem in [
            "ANP",
            "CANP",
            "ConvCNP",
            "TNPD",
        ]:
            continue
        # need to expand internal grid of ConvCNP
        if path.stem == "ConvCNP":
            if cfg.data.name == "128x128":
                m_cfg.model.kwargs.s_lower = [-4.5, -4.5]
                m_cfg.model.kwargs.s_upper = [4.5, 4.5]
            elif cfg.data.name == "256x256":
                m_cfg.model.kwargs.s_lower = [-8.5, -8.5]
                m_cfg.model.kwargs.s_upper = [8.5, 8.5]
            elif cfg.data.name == "1024x1024":
                m_cfg.model.kwargs.s_lower = [-32.5, -32.5]
                m_cfg.model.kwargs.s_upper = [32.5, 32.5]
        model = instantiate(m_cfg.model)
        if path.stem == "ConvCNP":
            print(model)
            state = TrainState.create(
                apply_fn=model.apply,
                params=state.params,
                kwargs=state.kwargs,
                tx=state.tx,
            )
        batches = dataloader(rng_data)
        batch = next(batches)
        try:
            model.valid_step(rng, state, batch)  # precompile
        except Exception:
            with open(out_fn.with_suffix(".txt"), "w") as f:
                f.write("OOM\n")
            continue
        pbar = tqdm(
            range(cfg.valid_num_steps),
            unit=" samples",
            leave=False,
            dynamic_ncols=True,
        )
        metrics = defaultdict(list)
        for i in pbar:
            rng_i, rng = random.split(rng)
            batch = next(batches)
            start = time()
            m = model.valid_step(rng_i, state, batch)
            end = time()
            m["s_elapsed"] = end - start
            for k, v in m.items():
                metrics[k] += [float(v)]
        with open(out_fn.with_suffix(".json"), "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
