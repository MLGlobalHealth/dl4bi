#!/usr/bin/env python3
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
import optax
import wandb
from jax import random
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid
import json
import networkx as nx

from dl4bi.meta_learning.train_utils import (
    Callback,
    cfg_to_run_name,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    log_graph_plots,
    save_ckpt,
    select_steps,
    train,
)


@hydra.main("configs/outbreaks", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    train_dataloader = build_dataloader(cfg.data_path, cfg.file_name, cfg.graph_dist, cfg.batch_size)
    valid_dataloader = build_dataloader(cfg.data_path, cfg.file_name, cfg.graph_dist, cfg.batch_size)
    lr_schedule = cosine_annealing_lr(
        cfg.train_num_steps,
        cfg.lr_peak,
        cfg.lr_pct_warmup,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.clip_max_norm),
        optax.yogi(lr_schedule),
    )
    model = instantiate(cfg.model)
    train_step, valid_step = select_steps(model)
    cmap = mpl.colormaps.get_cmap("grey")
    cmap.set_bad("blue")
    norm = mpl.colors.Normalize(vmin=cfg.vmin, vmax=cfg.vmax, clip=True)
    
    with open(cfg.data_path + 'graph_pos.json', 'r') as infile:
            pos = json.load(infile)
    graph = nx.read_adjlist(cfg.data_path + 'graph.adjlist')
    
    img_cbk = Callback(
        partial(log_graph_plots, pos=pos, graph=graph, norm=norm, vmin=cfg.vmin, vmax=cfg.vmax),
        cfg.plot_interval,
    )
    state = train(
        rng_train,
        model,
        optimizer,
        train_step,
        valid_step,
        train_dataloader,
        valid_dataloader,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[img_cbk],
    )
    metrics = evaluate(
        rng_test,
        state,
        valid_step,
        valid_dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloader(
    data_path: str,
    file_name: str,
    graph_dist_path: str,
    batch_size: int = 16,
    min_ctx_valid_pct: float = 0.05,
    max_ctx_valid_pct: float = 0.5,
):
    path = data_path + file_name  # contains [time, f_test]
    dataset = np.load(path, mmap_mode="r")['outbreaks']
    dataset = dataset[:, 1:]  # remove sim_id
    B = batch_size
    L = dataset.shape[1] - 1
    num_ctx_min = int(L * min_ctx_valid_pct)
    num_ctx_max = int(L * max_ctx_valid_pct)
    num_test_max = L
    
    graph_dist_path = data_path + graph_dist_path
    graph_dist = jnp.load(graph_dist_path)
    
    # TODO: better embedding/representation of nodes
    # node_embedding = build_grid([dict(start=-2.0, stop=2.0, num=16)] * 2).reshape(L, 2)
    node_embedding = build_grid([dict(start=-2.0, stop=2.0, num=int(np.ceil(np.sqrt(L))))] * 2).reshape(-1,1)[:L* 2].reshape(L, 2)
    # node_embedding = np.load(data_path + "node_embeddings.npy")
    # assert node_embedding.shape[0] == L
    # print("Node Embeddings:\n", node_embedding)
    s_grid = jnp.repeat(node_embedding[None, ...], B, axis=0)  # [L, 2] -> [B, L, 2]
    valid_lens_test = jnp.repeat(num_test_max, B)
    N = dataset.shape[0]

    def dataloader(rng: jax.Array):
        while True:
            rng_batch, rng_permute, rng_valid, rng = random.split(rng, 4)
            batch_idx = random.choice(rng_batch, N, (B,), replace=False)
            permute_idx = random.choice(rng_permute, L, (L,), replace=False)
            # permute_idx = jnp.arange(L)
            batch = dataset[batch_idx]
            time, f_test = batch[:, [0]], batch[:, 1:]
            time = jnp.repeat(time[:, None, :], L, axis=1)
            f_test = 2 * (f_test - 0.5)  # [0, 1] -> [-1, 1]
            s_test = jnp.concatenate([s_grid, time], axis=-1)
            f_test = f_test.reshape(B, -1, 1)  # [B, H, W, 1] -> [B, L, 1]
            inv_permute_idx = jnp.argsort(permute_idx)
            # permute the order and select the first valid_lens_ctx for context
            s_test_permuted = s_test[:, permute_idx, :]
            f_test_permuted = f_test[:, permute_idx, :]
            s_test_permuted = s_test_permuted[:, :num_test_max, :]
            f_test_permuted = f_test_permuted[:, :num_test_max, :]
            valid_lens_ctx = random.randint(
                rng_valid,
                (B,),
                num_ctx_min,
                num_ctx_max,
            )
            yield (
                s_test_permuted,  # s_ctx (permuted)
                f_test_permuted,  # f_ctx (permuted)
                valid_lens_ctx,  # only the first valid lens are used/observed
                s_test_permuted,  # s_test (permuted)
                f_test_permuted,  # f_test (permuted)
                valid_lens_test,
                s_test,  # add full originals for use in callbacks, e.g. log_plots
                f_test,
                (inv_permute_idx, inv_permute_idx),
                graph_dist,
            )

    return dataloader


if __name__ == "__main__":
    main()
