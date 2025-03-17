#!/usr/bin/env python3
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import wandb
from jax import jit, random
from jax.scipy import stats
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from sps.utils import build_grid
from jax.scipy.stats import norm

from dl4bi.core import mask_from_valid_lens
from dl4bi.meta_regression.train_utils import (
    TrainState,
    cfg_to_run_name,
    load_ckpt,
    log_wandb_line,
    plot_posterior_predictive,
)


# NOTE: use the same configs as the Outbreaks models
@hydra.main("configs/outbreaks", config_name="default", version_base=None)
def main(cfg: DictConfig):
    project_parent = cfg.get("project_parent")
    if re.match(".*Outbreaks.*", cfg.project, re.IGNORECASE):
        project_parent = project_parent or cfg.project
        cfg.project = "Active Learning"
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    cfg.batch_size = 1
    print(OmegaConf.to_yaml(cfg))
    num_tasks, budget, num_init = 10, 90, 10
    rng = random.key(cfg.seed)
    _, rng_data, rng_opt = random.split(rng, num=3) # TODO: check 
    # dataloader = build_dataloader(cfg.data_path, cfg.file_name, cfg.graph_dist, cfg.temporal_data, cfg.batch_size) # TODO: add test_time
    dataloader = build_al_dataloader(cfg.data_path, cfg.test_file_name, num_tasks, budget, num_init)
    
    rng_data, rng_permute = random.split(rng)
    batches = dataloader(rng_data)
    s_tests, f_tests = [], []
    for i in tqdm(range(num_init+budget), desc="Building dataset"):
        s_test, f_test, inv_permute_idx_test = next(batches)
        # Note: inv_permute_idx_test is the same for different time steps
        s_tests += [s_test]
        f_tests += [f_test]
    s_test, f_test = jnp.vstack(s_tests), jnp.vstack(f_tests)
    D_s, D_f = s_test.shape[-1], f_test.shape[-1]
    s_test = s_test.reshape(num_init + budget, num_tasks, -1, D_s) # [T, B, L, D_s=3]
    f_test = f_test.reshape(num_init + budget, num_tasks, -1, D_f) # [T, B, L, D_f=1]
    s_test = jnp.transpose(s_test, (1, 0, 2, 3)) #.reshape(num_tasks, -1, D_s) # [B, T * L, D_s]
    f_test = jnp.transpose(f_test, (1, 0, 2, 3)) #.reshape(num_tasks, -1, D_f) # [B, T * L, D_f]
    print('s_test shape:', s_test.shape)
    print('f_test shape:', f_test.shape)
    
    # load model
    path = Path(f"results/{project_parent}/{cfg.seed}/{run_name}")
    model_state, _ = load_ckpt(path.with_suffix(".ckpt"))
    model_fn = jit_model_fn(model_state)
    
    # load graph
    graph_dist_path = cfg.data_path + cfg.graph_dist
    graph_dist = jnp.load(graph_dist_path)
        
    loss= optimize(
        rng_opt, s_test, f_test,  graph_dist, inv_permute_idx_test, model_fn, num_init, budget
    )
    for k, v in loss.items():
        v = jnp.array(v)
        v = v.transpose()
        print(f'{k} shape:', v.shape)
        log_wandb_line(v.mean(axis=0), k + " mu")
        log_wandb_line(v.std(axis=0), k + " std")
        log_regret_dist(v, k)
    wandb.finish()


def jit_model_fn(state: TrainState):
    @jit
    def model_fn(
        s_ctx: jax.Array,
        f_ctx: jax.Array,
        s_test: jax.Array,
        valid_lens_ctx: jax.Array,
        valid_lens_test: jax.Array,
        inv_permute_idx: jax.Array,
        inv_permute_idx_test: jax.Array,
        graph_dist: jax.Array,
        rng_extra: jax.Array,
    ):
        return state.apply_fn(
            {"params": state.params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            valid_lens_test,
            inv_permute_idx=inv_permute_idx,
            inv_permute_idx_test=inv_permute_idx_test,
            graph_dist=graph_dist,
            rngs={"extra": rng_extra},
        )

    return model_fn

def build_al_dataloader(
    data_path: str,
    file_name: str,
    num_tasks: int,
    budget: int,
    num_init: int,
):
    """
    Build a dataloader for active learning experiments,
    where generates s_test and f_test from the same simulation_id, with time in order.
    """
    path = data_path +  file_name # contains [time, f_test]
    dataset = np.load(path, mmap_mode="r")['outbreaks'] # [sim_id, time, f_test]
    print('dataset shape:', dataset.shape)

    # only remain num_init + budget time steps for each simulation
    dataset = dataset[dataset[:, 1] < num_init + budget] 
    # check for each sim_id, there are at least num_init + budget time steps
    sim_ids, counts = np.unique(dataset[:, 0], return_counts=True)
    assert np.all(counts == num_init + budget)
    num_sims = len(sim_ids)
    time_steps = num_init + budget
    dataset = dataset.reshape(num_sims, time_steps, -1) # [num_sims, time_steps, num_nodes + 2]; 2 for sim_id and time
    print('dataset shape:', dataset.shape)
    B = num_tasks
    L = dataset.shape[-1] - 2
    
    
    s_grid = build_grid([dict(start=-2.0, stop=2.0, num=int(np.ceil(np.sqrt(L))))] * 2).reshape(-1,1)[:L* 2].reshape(L, 2)
    s_grid = jnp.repeat(s_grid[None, ...], B, axis=0)  # [L, 3] -> [B, L, 3]
    
    selected_sim_ids = random.choice(random.PRNGKey(0), num_sims, (B,), replace=False)
    # print('selected_sim_ids:', selected_sim_ids)

    def dataloader(rng: jax.Array, current_time=0):
        while True:
            print('current time:', current_time)
            print('selected_sim_ids:', selected_sim_ids)
            batch = dataset[selected_sim_ids, current_time, :]
            time, f_test = batch[:, [1]], batch[:, 2:]
            time = jnp.repeat(time[:, None, :], L, axis=1)
            # f_test = 2 * (f_test - 0.5)  # [0, 1] -> [-1, 1]
            s_test = jnp.concatenate([s_grid, time], axis=-1)
            f_test = f_test.reshape(B, -1, 1)  # [B, H, W, 1] -> [B, L, 1]
            current_time += 1
            permute_idx_test = random.permutation(rng, L)
            # debug: identity permutation
            # permute_idx_test = jnp.arange(L)
            inv_permute_idx_test = jnp.argsort(permute_idx_test)
            # TODO: permutation? 
            yield (
                s_test[:,permute_idx_test,:],  # s_ctx (permuted over nodes)
                f_test[:,permute_idx_test,:],  # f_ctx (permuted over nodes)
                inv_permute_idx_test,
            )
    return dataloader     

def optimize(
    rng: jax.Array,
    s_test: jax.Array,  # [B, T, L, D_s=3]
    f_test: jax.Array,  # [B, T, L, D_f=1]
    graph_dist: jax.Array,
    inv_permute_idx_test: jax.Array,
    model_fn: Callable,
    num_init: int = 1,
    budget: int = 50,
):
    (B, T, L, D_s), T_ctx = s_test.shape, num_init + budget
    # s_ctx = jnp.zeros((B, T_ctx, L, D_s))
    # f_ctx = jnp.zeros((B, T_ctx, L, 1))
    # s_ctx = s_ctx.at[:, :num_init, :, :].set(s_test[:, :num_init, :, :])
    # f_ctx = f_ctx.at[:, :num_init, :, :].set(f_test[:, :num_init, :, :])
    s_ctx = s_test[:, :num_init, :, :]
    f_ctx = f_test[:, :num_init, :, :]
    
    valid_lens_ctx = jnp.repeat(int(0.1 * L * num_init), B)  # 10% of nodes for num_init time steps are observed
    valid_lens_test = jnp.repeat(L, B)
    rng_extra, rng = random.split(rng)
    
    inv_permute_idx_ctx = jnp.repeat(jnp.repeat(inv_permute_idx_test, num_init), B).reshape(B, -1)
    inv_permute_idx_test = jnp.repeat(inv_permute_idx_test, B).reshape(B, -1)
    
    loss = {"NLL": [], "RMSE": [], "MAE": [], "Coverage": []}
    
    for i in tqdm(range(budget), desc="Optimizing"):
        s_ctx = s_ctx.reshape(B, -1, D_s)
        f_ctx = f_ctx.reshape(B, -1, 1)
        
        s_test_i = s_test[:,i + num_init,:,:].reshape(B, -1, D_s)
        
        print('s_ctx shape:', s_ctx.shape)
        print('f_ctx shape:', f_ctx.shape)
        print('s_test_fit shape:', s_test_i.shape)
        print('valid_lens_ctx shape:', valid_lens_ctx.shape)
        print('valid_lens_test shape:', valid_lens_test.shape)
        print('inv_permute_idx shape:', inv_permute_idx_ctx.shape)
        print('inv_permute_idx_test shape:', inv_permute_idx_test.shape)
        
        f_mu, f_std, *_ = model_fn(s_ctx, f_ctx, s_test_i, valid_lens_ctx, valid_lens_test, inv_permute_idx_ctx, inv_permute_idx_test, graph_dist, rng_extra)
        print('f_mu shape:', f_mu.shape)
        print('f_std shape:', f_std.shape)
        # print('f_mu:', f_mu)
        # print('f_std:', f_std)
        # argmax_f_std = jnp.argsort(f_std, axis=1)[:, -int(0.1 * L):]
        # max_f_std = jnp.take_along_axis(f_std, argmax_f_std, axis=1)
        # # print("max_f_std shape: ", max_f_std.shape)
        # print("max_f_std: ", max_f_std)
        
        
        # e = node_entropy(f_std).reshape(B, -1)
        # e = random_policy(f_std).reshape(B, -1)
        e = local_entropy(f_std, graph_dist).reshape(B, -1)
        # pick the 10% nodes with the highest entropy
        # permuate s_test_i so that the idx in argmax_e is at the front (observed)
        permute_idx_al = jnp.argsort(e, axis=1)
        inv_permute_idx_al = jnp.argsort(permute_idx_al, axis=1)
        
        argmax_e = permute_idx_al[:, -int(0.1 * L):]
        max_e = jnp.take_along_axis(e, argmax_e, axis=1)
        print("max_e shape: ", max_e.shape)
        
        # print("max_e: ", max_e)
        print('permute_idx_al:', permute_idx_al.shape)
        print('inv_permute_idx_al', inv_permute_idx_al.shape)
        
        print('permute_idx_al:', permute_idx_al)
        print('inv_permute_idx_al', inv_permute_idx_al)
        
        
        
        s_ctx = jnp.concatenate([s_ctx[:,L:,:], s_test_i], axis=1)
        f_test_i = f_test[:,i + num_init,:,:].reshape(B, -1, 1)
        f_ctx = jnp.concatenate([f_ctx[:,L:,:], f_test_i], axis=1)
        inv_permute_idx_ctx = jnp.concatenate([inv_permute_idx_ctx[:,L:], inv_permute_idx_al], axis=1) # todo: bug here, check inv_permute_idx_ctx[:,L:] length
        valid_lens_ctx = jnp.repeat(int(0.1 * L * (num_init)), B)
        
        loss = evaluate_al_t(i, loss, f_mu, f_std, f_test_i)
    return loss 

def node_entropy(f_std: jax.Array):
    # f_std: [B, L, 1]
    entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * f_std ** 2)
    print('entropy shape:', entropy.shape)
    # print('entropy:', entropy)
    return entropy

def local_entropy(f_std: jax.Array, graph_dist: jax.Array):
    # f_std: [B, L, 1]
    # for each node, calculate the weighted entropy of its neighbors, and itself
    # weighted by the inverse graph distance
    ne = node_entropy(f_std)
    entropy = jnp.sum(ne[:, None, :] / graph_dist[None, :, :], axis=-1)
    # entropy = jnp.zeros_like(ne)
    # for i in range(f_std.shape[1]):
    #     for j in range(f_std.shape[1]):
    #         entropy = entropy.at[:, i].add(ne[:, j] / graph_dist[i, j])
    return entropy

def random_policy(f_std: jax.Array):
    rng = random.PRNGKey(0)
    return random.uniform(rng, f_std.shape)


def evaluate_al_t(
    t: int,
    loss: dict,
    f_mu: jax.Array,
    f_std: jax.Array,
    f_test: jax.Array,
    hdi_prob: float = 0.95,
    ):
    # f_mu: [B, L, 1]
    # f_std: [B, L, 1]
    # f_test: [B, L, 1]
    print('inside evaluate_al_t')
    print('f_mu shape:', f_mu.shape)
    print('f_std shape:', f_std.shape)
    print('f_test shape:', f_test.shape)
    B = f_mu.shape[0]
    nll = -norm.logpdf(f_test, f_mu, f_std).mean(axis=1).reshape(B,) # mean over nodes
    rmse = jnp.sqrt(jnp.square(f_test - f_mu).mean(axis=1)).reshape(B,)
    mae = jnp.abs(f_test - f_mu).mean(axis=1).reshape(B,)
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    f_lower, f_upper = f_mu - z_score * f_std, f_mu + z_score * f_std
    cvg = ((f_test >= f_lower) & (f_test <= f_upper)).mean(axis=1).reshape(B,)
    print('nll:', nll)
    print('rmse:', rmse)
    print('mae:', mae)
    print('cvg:', cvg)
    
    loss["NLL"] += [nll]
    loss["RMSE"] += [rmse]
    loss["MAE"] += [mae]
    loss["Coverage"] += [cvg]
    # loss = {m: np.mean(vs) for m, vs in loss.items()}  # average over batches
    
    # debug
    paths_scatter = []
    plt.scatter(f_mu[0], f_test[0], c='b')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title('Prediction vs Ground Truth')
    paths_scatter += [f"/tmp/{datetime.now().isoformat()} - sample {t} - scatter.png"]
    plt.savefig(paths_scatter[-1], dpi=125)
    plt.clf()
    plt.close()
    wandb.log({f"Step {t} - scatter": [wandb.Image(p) for p in paths_scatter]}, step=t)
    
    return loss

def log_regret_dist(regret: jax.Array, eva_name: str, hdi_prob: float = 0.95):
    mu, std = regret.mean(axis=0), regret.std(axis=0)
    z = jnp.abs(stats.norm.ppf((1 - hdi_prob) / 2))
    iter = jnp.arange(regret.shape[1])
    plt.plot(iter, mu, color="black")
    plt.fill_between(
        iter,
        mu - z * std,
        mu + z * std,
        color="steelblue",
        alpha=0.4,
        interpolate=True,
    )
    ax = plt.gca()
    ax.set_xlabel("Iteration")
    ax.set_ylabel(eva_name)
    path = f"/tmp/{datetime.now().isoformat()}{eva_name}_dist.png"
    plt.title("Performance Distribution")
    plt.savefig(path, dpi=125)
    plt.clf()
    wandb.log({"Performance Distribution": wandb.Image(path)})
    return path

if __name__ == "__main__":
    main()
