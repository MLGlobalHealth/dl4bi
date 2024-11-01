from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from jax import jit, lax, random, vmap
from jax.tree_util import Partial
import networkx as nx


@jit
def mask_attn(x: jax.Array, valid_lens: jax.Array, fill=-jnp.inf):
    r"""Mask `x` with `fill` using `valid_lens`.

    Args:
        x: Values of dimension $\mathbb{R}^{B\times Q\times K}$
        valid_lens: Mask consisting of valid length per sequence
            $\mathbb{R}^{B}$ or $\mathbb{R}^{B\times Q}.

    Returns:
       `x` with filled values according to mask.
    """
    B, Q, K = x.shape
    if valid_lens.ndim == 1:
        valid_lens = jnp.repeat(valid_lens, Q)
    x = x.reshape(B * Q, K)
    m = jnp.arange(K) < valid_lens.reshape(-1, 1)
    # visualize_graph(m.reshape(B, Q, K)[0,:,:], 'Mask_Attn[0]')
    return jnp.where(m, x, fill).reshape(B, Q, K)

def load_adj_list(file_path):
    graph = {}  # Dictionary to store adjacency list
    with open(file_path, 'r') as file:
        for neighbors in file:
            # Convert neighbors to a list of integers
            if '#' not in neighbors:
                node_list = [int(n) for n in neighbors.strip().split(' ')]
                # Add node and its neighbors to the graph
                curr_node = node_list[0]
                if curr_node not in graph:
                    graph[curr_node] = node_list[1:]
                else:
                    graph[curr_node] += node_list[1:]
                for neighbor in node_list[1:]:
                    if neighbor not in graph:
                        graph[neighbor] = [curr_node]
                    else:
                        graph[neighbor].append(curr_node)
    return graph

def convert_graph_to_mask(graph, inv_permute_idx=None):
    L = len(graph)
    # if inv_permute_idx is None:
    #     permute_idx = jnp.arange(L)
    # else:
    permute_idx = inv_permute_idx.argsort()
    mask = jnp.zeros((L, L), dtype=bool)
    
    # num_edges = sum(len(neighbors) for neighbors in graph.values())
    # print(f'Number of edges: {num_edges}') # 960
    
    for node in graph:
        mask = mask.at[node, node].set(True)
        mask = mask.at[node, graph[node]].set(True)
    mask = mask[permute_idx, :][:, permute_idx]
    
    # visualize_graph(mask, 'Mask_Adj')
    
    # Set some random indices of mask to be True to decrease its sparsity
    # rng = random.PRNGKey(0)
    # num_random_entries = L * L // 10  # Number of random entries to set to True
    # random_indices = random.randint(rng, (num_random_entries, 2), 0, L)
    # mask = mask.at[random_indices[:, 0], random_indices[:, 1]].set(True)
    
    # if inv_permute_idx is not None:
    #     mask_inv = mask[inv_permute_idx, :][:, inv_permute_idx]
    # visualize_graph(mask, 'Mask_Adj_Random')
    # visualize_graph(mask_inv, 'Mask_adj_permuted')
    
    return mask
    
import matplotlib.pyplot as plt

def visualize_graph(matrix, name='Adjacency_Matrix'):
    plt.imshow(np.array(matrix), cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title(name)
    plt.savefig('/home/scratch/menang/outbreaks/' + name + '.png')
    plt.clf()

# @jit
def mask_attn_graph(x: jax.Array, inv_permute_idx, fill=-jnp.inf):
    r"""Mask `x` with `fill` using adjancency matrix from graph.
    
    Args: 
        x: Values of dimension $\mathbb{R}^{B\times Q\times K}$
        
    Returns:
        `x` with filled values according to mask.
    """
    adj_matrix_path = '/home/scratch/menang/outbreaks/dim16_lattice.adjilist'
    graph = load_adj_list(adj_matrix_path)
    mask = convert_graph_to_mask(graph, inv_permute_idx)
    B, Q, K = x.shape
    mask = jnp.broadcast_to(mask, (B, Q, K))
    x = jnp.where(mask, x, fill)
    return x

def mask_from_valid_lens(max_len: int, valid_lens: jax.Array):
    """Return a boolean mask using `valid_lens`.

    .. note::
        Adds a final dimension of 1, which is often used to broadcast across the
        final tensor dimension.
    """
    return (jnp.arange(max_len) < valid_lens[..., None])[..., None]


def pad_concat(x: jax.Array, y: jax.Array):
    """Concat channels of two layers, padding smaller spatial layer with zeros.

    Args:
        x: Tensor of shape [B, L_x, C_x].
        y: Tensor of shape [B, L_y, C_y].

    Returns:
        A concatenated tensor of shape [B, max(L_x, L_y), C_x + C_y].
    """
    L_x, L_y = x.shape[1], y.shape[1]
    padding = np.abs(L_x - L_y)
    is_even = padding % 2 == 0
    half_even = (padding // 2, padding // 2)
    half_odd = ((padding - 1) // 2, (padding + 1) // 2)
    p_e = Partial(jnp.pad, pad_width=((0, 0), half_even, (0, 0)), mode="reflect")
    p_o = Partial(jnp.pad, pad_width=((0, 0), half_odd, (0, 0)), mode="reflect")
    if L_x > L_y:
        y = p_e(y) if is_even else p_o(y)
    elif L_y > L_x:
        x = p_e(x) if is_even else p_o(x)
    return jnp.concatenate([x, y], axis=-1)


@partial(jit, static_argnames=("num_samples"))
def bootstrap(
    rng: jax.Array,
    x: jax.Array,  # [B, L, D]
    valid_lens: jax.Array,  # [B]
    num_samples: int = 1,
):
    """Bootstrap selects the first `valid_lens` values of `x` `num_samples` times.

    Args:
        rng: A PRNGKey.
        x: Array to bootstrap.
        valid_lens: The valid entries for every sequence in x.

    Returns:
        A bootstrap sampled array of shape [B * num_samples, L, D].
    """
    (B, L, _), K = x.shape, num_samples
    x = jnp.repeat(x, K, axis=0)
    valid_lens = jnp.repeat(valid_lens, K, axis=0)
    mask = mask_from_valid_lens(L, valid_lens).squeeze()
    rnd_idx = random.randint(rng, (B * K, L), 0, valid_lens[:, None])
    ord_idx = jnp.repeat(jnp.arange(L)[None, :], B * K, axis=0)
    boot_idx = mask * rnd_idx + ~mask * ord_idx
    return vmap(lambda row, idx: row[idx], (0, 0))(x, boot_idx), valid_lens


def breakpoint_if_nonfinite(x):
    """Create a breakpoint when non-finite values in `x`."""
    is_finite = jnp.isfinite(x).all()

    def true_fn(x):
        pass

    def false_fn(x):
        jax.debug.breakpoint()

    lax.cond(is_finite, true_fn, false_fn, x)
