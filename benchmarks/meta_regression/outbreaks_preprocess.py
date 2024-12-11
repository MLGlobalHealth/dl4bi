import numpy as np
import pandas as pd
import jax.numpy as jnp
from collections import deque
import os
import requests
import tarfile
import matplotlib.pyplot as plt
import networkx as nx
import json
import imageio
import jax
import jraph
from flax import linen as nn
from node2vec import Node2Vec

def load_adj_matrix(save_path):
    # load the graph
    graph = {}  # Dictionary to store adjacency list
    
    with open(save_path + 'graph.adjlist', 'r') as file:
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
    
    L = len(graph)                  
    adj_matrix = jnp.zeros((L, L), dtype=bool)
    for node in graph:
        adj_matrix = adj_matrix.at[node, node].set(True)
        adj_matrix = adj_matrix.at[node, graph[node]].set(True)
    with open(save_path + 'adj_matrix.npy', 'wb') as f:
        np.save(f, adj_matrix)
    
    return adj_matrix

def bfs_distance(adj_matrix, start_node):
    n = len(adj_matrix)
    distances = [float('inf')] * n
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current = queue.popleft()
        for neighbor, is_connected in enumerate(adj_matrix[current]):
            if is_connected and distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    return distances

def calculate_distances_normalized(adj_matrix, save_path):
    n = len(adj_matrix)
    distances = np.full((n, n), float('inf'))  # Initialize with 'inf'

    distances = np.array([bfs_distance(adj_matrix, i) for i in range(n)])
    assert np.all(distances == distances.T), "The distances matrix should be symmetric."
    assert np.isfinite(distances).all(), "All distances should be finite."

    # Normalize the distances between 0 and 1
    finite_distances = distances[np.isfinite(distances)]
    min_distance = np.min(finite_distances)
    max_distance = np.max(finite_distances)

    # Avoid division by zero in case all distances are the same
    if max_distance > min_distance:
        distances = (distances - min_distance) / (max_distance - min_distance)
    else:
        distances = np.zeros_like(distances)  # If all distances are the same, set to 0
        
    return distances

def bfs_distance_upper_triangle(adj_matrix, start_node):
    n = len(adj_matrix)
    distances = [float('inf')] * n
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current = queue.popleft()
        current_distance = distances[current]  # Cache the current distance
        # Only iterate over neighbors in the upper triangle
        for neighbor in range(current + 1, n):
            if adj_matrix[current][neighbor] and distances[neighbor] == float('inf'):
                distances[neighbor] = current_distance + 1
                queue.append(neighbor)
    
    return distances

def calculate_distances_upper_triangle_normalized(adj_matrix, save_path):
    n = len(adj_matrix)
    distances = np.full((n, n), float('inf'))  # Initialize with 'inf'

    # Only calculate distances for the upper triangle
    for i in range(n):
        bfs_result = bfs_distance_upper_triangle(adj_matrix, i)
        for j in range(i, n):  # Only fill the upper triangle
            distances[i][j] = bfs_result[j]

    # Mirror the upper triangle to the lower triangle
    for i in range(n):
        for j in range(i + 1, n):
            distances[j][i] = distances[i][j]

    # Normalize the distances between 0 and 1
    finite_distances = distances[np.isfinite(distances)]
    min_distance = np.min(finite_distances)
    max_distance = np.max(finite_distances)

    # Avoid division by zero in case all distances are the same
    if max_distance > min_distance:
        distances = (distances - min_distance) / (max_distance - min_distance)
    else:
        distances = np.zeros_like(distances)  # If all distances are the same, set to 0
    
    visualize_matrix(distances, 'Distances_Matrix_UT_infinite')
    # Set infinite distances to 1
    distances[np.isinf(distances)] = 1
    visualize_matrix(distances, 'Distances_Matrix_UT')

    # Save the distances as a binary .npy file
    # np.save(save_path + 'distances.npy', distances)
    
    return distances

def distance_mask(distances, threshold=0.8):
    mask = distances > threshold
    distances[mask] = 1
    return distances
    


def visualize_matrix(matrix, save_path, name='Adjacency_Matrix'):
    plt.imshow(np.array(matrix), cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title(name)
    plt.savefig(save_path + name + '.png')
    plt.clf()
    
def visualize_graph(adjlist_path, save_path, outbreaks=None, step='0'):
    # read in graph
    graph = nx.read_adjlist(adjlist_path)
    # map node names to integers
    mapping = { node: int(node) for node in graph.nodes() }
    graph = nx.relabel_nodes(graph, mapping)
    
    if os.path.exists(save_path + 'graph_pos.json'):
        with open(save_path + 'graph_pos.json', 'r') as infile:
            pos = json.load(infile)
    else:    
        # generate node positions and visualise
        pos = nx.spring_layout(graph, seed=33)
        pos_export = { k: list(v) for k, v in pos.items() }
        with open(save_path + 'graph_pos.json', 'w+') as outfile:
            outfile.write(json.dumps(pos_export))
    
    if outbreaks is None:
        nx.draw(graph, pos, with_labels=False, node_size=50)
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        pos = {int(k): v for k, v in pos.items()}  # Ensure positions are correctly assigned to nodes
        nx.draw(graph, pos, with_labels=False, node_size=30, ax=ax)
        nx.draw_networkx_nodes(graph, pos, node_size=30, node_color=outbreaks, cmap='bwr', ax=ax, vmin=0, vmax=1)
        plt.colorbar(plt.cm.ScalarMappable(cmap='bwr'), ax=ax)
    plt.title('Graph with Outbreaks at Step ' + step)
    plt.savefig(save_path + 'graph_' + step + '.png')
    
def vis_ave_outbreaks(outbreaks, steps, path):
    # Take average of outbreaks where the first column equals to 0
    
    for step in steps:
        filtered_outbreaks = outbreaks[outbreaks[:, 0] == step]
        if filtered_outbreaks.size > 0:
            average_outbreaks = np.mean(filtered_outbreaks, axis=0)
            average_outbreaks = average_outbreaks[1:] #.reshape(16, 16)
            # print("Average of filtered outbreaks:", average_outbreaks)
            # visualize_matrix(average_outbreaks, 'Average_Outbreaks')
            visualize_graph(path + 'graph.adjlist', path, outbreaks=average_outbreaks, step=str(step))
        else:
            print("No outbreaks with the first column equal to ", step) 
            
    if len(steps) > 1:
        # Create a GIF of the average outbreaks over the steps
        images = []
        for step in steps:
            image_path = f'{path}graph_{step}.png'
            if os.path.exists(image_path):
                images.append(imageio.imread(image_path))
        
        gif_path = path + 'graph_outbreaks.gif'
        for step in steps:
            image_path = f'{path}graph_{step}.png'
            if os.path.exists(image_path):
                os.remove(image_path)
        imageio.mimsave(gif_path, images, duration=1)
        print(f"GIF saved at {gif_path}")
        
def load_simulations(path):
    # Load the simulation data   
    if os.path.exists(path + 'outbreaks.npz'):
        outbreaks = np.load(path + 'outbreaks.npz')['outbreaks']
        outbreaks_with_sim_num = None
    elif os.path.exists(path + 'outbreaks.npy'):
        outbreaks = np.load(path + 'outbreaks.npy')
        outbreaks_with_sim_num = None
    elif os.path.exists(path + 'outbreaks.csv'):
        outbreaks_with_sim_num = pd.read_csv(path + 'outbreaks.csv').to_numpy()
        outbreaks = outbreaks_with_sim_num[:,1:].astype(np.int32)
    else:
        raise ValueError("Invalid file format. Please provide a .npy or .csv file.") 
    
    if 'SB_high' in path:
        step  = 60
    else:
        step = 100
    outbreaks = outbreaks[outbreaks[:, 0] <=step]
    np.savez_compressed(path + 'outbreaks.npz', outbreaks=outbreaks)   
    
    print("Outbreaks:", outbreaks)
    print("Outbreaks shape:", outbreaks.shape)
    print("Outbreaks time steps:", outbreaks[:1000, 0])
    
    
    # vis_ave_outbreaks(outbreaks, 0)
    if outbreaks_with_sim_num is not None:
        vis_ave_outbreaks(outbreaks_with_sim_num[outbreaks_with_sim_num[:, 0] == 1][:,1:], list(range(0, step)), path)
    else:
        vis_ave_outbreaks(outbreaks[:step], list(range(0, step)), path)
    return outbreaks

def node2vec_features(adj_matrix, dimensions=8):
    # Create Node2Vec object
    graph = nx.from_numpy_array(adj_matrix)  # Convert to NetworkX graph
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=10, num_walks=100, workers=1)

    # Train the embeddings
    model = node2vec.fit(window=5, min_count=1)
    embeddings = np.array([model.wv[str(node)] for node in range(adj_matrix.shape[0])])
    return embeddings

def main():
    save_path = 'cache/outbreaks_SB_low/'
    
    if not os.path.exists(save_path + 'adj_matrix.npy'): 
        adj_matrix = load_adj_matrix(save_path)
    else:
        adj_matrix = np.load(save_path + 'adj_matrix.npy')
    print("Adjacency Matrix:", adj_matrix)
    visualize_matrix(adj_matrix, save_path, 'Adjacency_Matrix')
    
    if not os.path.exists(save_path + 'distances.npy'):
        distances = calculate_distances_normalized(adj_matrix, save_path)
        np.save(save_path + 'distances.npy', distances)
    else:
        distances = np.load(save_path + 'distances.npy')
    visualize_matrix(distances, save_path, 'Distances_Matrix')
        
    if not os.path.exists(save_path + 'distances_UT.npy'):
        # distances = calculate_distances_normalized(adj_matrix, save_path)
        distances_UT = calculate_distances_upper_triangle_normalized(adj_matrix, save_path)
        
        # Save the distances as a binary .npy file
        np.save(save_path + 'distances_UT.npy', distances_UT)
    else:
        distances_UT = np.load(save_path + 'distances_UT.npy')
    visualize_matrix(distances_UT, save_path,  'Distances_Matrix_UT')
    
    distance_masked_threshold = distance_mask(distances, threshold=0.5)
    visualize_matrix(distance_masked_threshold, save_path, 'Distances_Matrix_Threshold_Mask')
    np.save(save_path + 'distances_masked_threshold.npy', distance_masked_threshold)
    
    # Check the percentage of distances that are the same between distances_UT and distance_masked_threshold
    same_distances = np.isclose(distances_UT, distance_masked_threshold)
    percentage_same = np.sum(same_distances) / same_distances.size * 100
    print(f"Percentage of distances that are the same: {percentage_same:.2f}%")
    
    # node_embeddings = node2vec_features(adj_matrix)
    # print("Node2Vec Features:\n", node_embeddings)
    # print("Node2Vec Features shape:", node_embeddings.shape)
    # np.save(save_path + 'node_embeddings.npy', node_embeddings)
    
    # print('load simulations')
    # load_simulations(save_path)
    
    # x_idx = np.array([0, 3])
    # y_idx = np.array([1, 2])
    # distances = distances[x_idx, :][:, y_idx]
    # if np.isinf(distances).any():
    #     inf_indices = np.argwhere(np.isinf(distances))
    #     print("Warning: There are infinite values in the distances matrix at indices:", inf_indices)
    # else:
    #     print("All distances are finite.")
    
    # print("Distances:", distances)
    
if __name__ == "__main__":
    main()