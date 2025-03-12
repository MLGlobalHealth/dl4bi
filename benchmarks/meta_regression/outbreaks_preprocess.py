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
from tqdm import tqdm

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
    
def visualize_graph(adjlist_path, save_path, outbreaks=None, step='0', vmin=0, vmax=1, cmap='bwr'):
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
        # vmin, vmax = np.min(outbreaks), np.max(outbreaks)
        nx.draw(graph, pos, with_labels=False, node_size=30, ax=ax)
        nx.draw_networkx_nodes(graph, pos, node_size=30, node_color=outbreaks, cmap=cmap, ax=ax, vmin=vmin, vmax=vmax)
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)), ax=ax)
    plt.title('Graph with Outbreaks at Step ' + step)
    plt.savefig(save_path + 'graph_' + step + '.png')
    
def vis_ave_outbreaks(outbreaks, steps, path, outbreak_name, vmin=0, vmax=1, cmap='bwr'):
    # Take average of outbreaks where the first column equals to 0
    
    for step in steps:
        filtered_outbreaks = outbreaks[outbreaks[:, 0] == step]
        if filtered_outbreaks.size > 0:
            average_outbreaks = np.mean(filtered_outbreaks, axis=0)
            average_outbreaks = average_outbreaks[1:] #.reshape(16, 16)
            # print("Average of filtered outbreaks:", average_outbreaks)
            # visualize_matrix(average_outbreaks, 'Average_Outbreaks')
            visualize_graph(path + 'graph.adjlist', path, outbreaks=average_outbreaks, step=str(step), vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            print("No outbreaks with the first column equal to ", step) 
            
    if len(steps) > 1:
        # Create a GIF of the average outbreaks over the steps
        images = []
        for step in steps:
            image_path = f'{path}graph_{step}.png'
            if os.path.exists(image_path):
                images.append(imageio.imread(image_path))
        
        gif_path = path + outbreak_name +'_graph.gif'
        for step in steps:
            image_path = f'{path}graph_{step}.png'
            if os.path.exists(image_path):
                os.remove(image_path)
        imageio.mimsave(gif_path, images, duration=1)
        print(f"GIF saved at {gif_path}")
    
def vis_col_distribution(outbreaks, path, outbreak_file, use_step, vmin=0, vmax=1, cmap='bwr'):
    
    plt.hist(outbreaks[:, 0], bins=range(int(outbreaks[:, 0].min()), int(outbreaks[:, 0].max()) + 1), edgecolor='black')
    plt.xlabel('Simulation IDs')
    plt.ylabel('Frequency')
    plt.title('Distribution of Simulation IDs in Outbreaks')
    plt.savefig(path + 'simulation_id_distribution.png')
    plt.clf()
    
    # Plot the distribution of the first column values in outbreaks
    plt.hist(outbreaks[:, 1], bins=range(int(outbreaks[:, 1].min()), int(outbreaks[:, 1].max()) + 1), edgecolor='black')
    plt.xlabel('Time steps')
    plt.ylabel('Frequency')
    plt.title('Distribution of Time Steps in Outbreaks')
    plt.savefig(path + 'time_distribution.png')
    plt.clf()
    
    # Plot the distribution of each column from the 3rd to the last column in outbreaks
    for col in np.random.choice(np.arange(2, outbreaks.shape[1]), 5, replace=False):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Original distribution
        axs[0].hist(outbreaks[:, col], bins=30, edgecolor='black')
        axs[0].set_xlabel(f'Column {col + 1} Values')
        axs[0].set_ylabel('Frequency')
        axs[0].set_title(f'Distribution of Column {col + 1} Values in Outbreaks')

        # Log-transformed distribution
        # from scipy.stats import boxcox
        # log_values = np.log1p(outbreaks[:, col])  # Use log1p to handle zero values
        # transformed_values, _ = boxcox(outbreaks[:, col] + 1)
        from sklearn.preprocessing import PowerTransformer
        power = PowerTransformer(method='yeo-johnson', standardize=True)
        transformed_values = power.fit_transform(outbreaks[:, col].reshape(-1, 1))
        axs[1].hist(transformed_values, bins=30, edgecolor='black')
        axs[1].set_xlabel(f'Power Transformation of Column {col + 1} Values')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title(f'Power Transformed Distribution of Column {col + 1} Values in Outbreaks')

        plt.tight_layout()
        plt.savefig(f'{path}column_{col + 1}_distribution.png')
        plt.clf()

    # Plot the distribution of the average of columns from the 3rd to the last column in outbreaks
    # average_values = np.mean(outbreaks[:, 2:], axis=0)
    # plt.hist(average_values, bins=30, edgecolor='black')
    # plt.xlabel('Average Values')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Average Values from in Outbreaks')
    # plt.savefig(path + 'average_labels_distribution.png')
    # plt.clf()
    
    unique_sim_ids = np.unique(outbreaks[:, 0])
    sampled_sim_ids = np.random.choice(unique_sim_ids, 5, replace=False)
    # sampled_sim_ids = [0, 1, 50000, 50001]
    print("Sampled Simulation IDs:", sampled_sim_ids)
    for vis_sim_id in sampled_sim_ids:
        vis_outbreaks = outbreaks[outbreaks[:, 0] == vis_sim_id][:, 1:]
        vis_ave_outbreaks(vis_outbreaks, list(range(0, use_step)), path, outbreak_file.split('.')[0] + '_sim_' + str(vis_sim_id), vmin=vmin, vmax=vmax, cmap=cmap)
    
def node2vec_features(adj_matrix, dimensions=8):
    # Create Node2Vec object
    graph = nx.from_numpy_array(adj_matrix)  # Convert to NetworkX graph
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=10, num_walks=100, workers=1)

    # Train the embeddings
    model = node2vec.fit(window=5, min_count=1)
    embeddings = np.array([model.wv[str(node)] for node in range(adj_matrix.shape[0])])
    return embeddings

def read_csvs(path, csv_file, save_path, save_name, filter_flag=False, use_step=99, normalise_population=None):
    # Decompress all tar.gz files under the specified path
    if not any(file_name.endswith('.csv') for file_name in os.listdir(path)):
        for file_name in os.listdir(path):
            if file_name.endswith('.tar.gz'):
                file_path = os.path.join(path, file_name)
                with tarfile.open(file_path, 'r:gz') as tar:
                    tar.extractall(path)
                print(f"Decompressed {file_name}")
            
    combined_df = pd.DataFrame()
    if csv_file is None:
        file_names = os.listdir(path)
    elif isinstance(csv_file, str):
        file_names = [csv_file]
    last_sim_id = 0
    for file_name in file_names:
        if file_name.endswith('.csv'):
            file_path = os.path.join(path, file_name)
            df = pd.read_csv(file_path)
            if df['sim_id'].iloc[0] < last_sim_id:
                df['sim_id'] = df['sim_id'] + last_sim_id + 1
            last_sim_id = df['sim_id'].iloc[-1] 
            print(df.shape)
            print(df)
            df = df[df['time'] <=use_step]
            print("after filtering largest time steps: ", df.shape)
            valid_sim_ids = df.groupby('sim_id')['time'].apply(lambda x: set(x) == set(range(use_step + 1)))
            valid_sim_ids = valid_sim_ids[valid_sim_ids].index
            print("valid sim ids: ", len(valid_sim_ids))
            df = df[df['sim_id'].isin(valid_sim_ids)]
    
            combined_df = pd.concat([combined_df, df], ignore_index=True, axis=0)
    if normalise_population is not None:
        combined_df.iloc[:, 2:] = combined_df.iloc[:, 2:] / normalise_population
        # from sklearn.preprocessing import MinMaxScaler, PowerTransformer
        # scaler = MinMaxScaler()
        # combined_df.iloc[:, 2:] = scaler.fit_transform(combined_df.iloc[:, 2:])
        # power = PowerTransformer(method='yeo-johnson', standardize=True)
        # combined_df.iloc[:, 2:] = power.fit_transform(combined_df.iloc[:, 2:])
        
    combined_df.iloc[:, 0] = combined_df.iloc[:, 0].astype(int)
    combined_df.iloc[:, 1] = combined_df.iloc[:, 1].astype(int)
            
    print("combine csvs: ", combined_df.shape)
    # if filter_flag:
    #     if 'sim_id' in combined_df.columns: 
    #         # Filter out rows with sim_id that only have one time=0 in the second column
    #         # sim_counts = combined_df.groupby('sim_id').size()
    #         # valid_sim_ids = sim_counts[sim_counts.iloc[:, 1] > 1][:,0].unique()
    #         # valid_sim_ids = sim_counts[sim_counts > use_step].index
    #         # print("sim ids: ", len(combined_df['sim_id'].unique()))
    #         # valid_sim_ids = combined_df[combined_df['time'] > use_step]['sim_id'].unique()
    #         print(combined_df.shape)
    #         combined_df = combined_df[combined_df['time'] <=use_step]
    #         print("after filtering largest time steps: ", combined_df.shape)
    #         valid_sim_ids = combined_df.groupby('sim_id')['time'].apply(lambda x: set(x) == set(range(use_step + 1)))
    #         valid_sim_ids = valid_sim_ids[valid_sim_ids].index
    #         print("valid sim ids: ", len(valid_sim_ids))
    #         combined_df = combined_df[combined_df['sim_id'].isin(valid_sim_ids)]
            
            
    #     else:
    #         raise ValueError("Invalid CSV file format. Please provide a CSV file with a 'sim_id' column.")
    #         # TODO
    #     print("with valid sim ids: ", combined_df.shape)
    #     combined_df = combined_df[combined_df['time'] <=use_step]
    #     print("after filtering: ", combined_df.shape)
    #     # Verify each simulation has exactly 0 to use_step time steps
    #     sim_time_counts = combined_df.groupby('sim_id')['time'].nunique()
    #     valid_sim_ids = sim_time_counts[sim_time_counts == (use_step + 1)].index
    #     combined_df = combined_df[combined_df['sim_id'].isin(valid_sim_ids)]
    #     print("after checking: ", combined_df.shape)
        
    combined_npz_path = os.path.join(save_path, save_name)
    np.savez_compressed(combined_npz_path, outbreaks=combined_df.to_numpy())
    print(f"Combined CSV data saved to {combined_npz_path}")

def load_distances(save_path):
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
    

def preprocess_SIR_continous_data(load_csv=False, graph='SB_high'):
    save_path = 'cache/outbreaks_' + graph + '/'
    load_csv_path = 'cache/synthetic-graphs/' + graph + '/'
    # load_csv_path = None
    file_name = None
    outbreak_file = 'SIR_outbreaks_continuous.npz'
    outbreak_path = save_path + outbreak_file
    vmin = 0
    vmax = 0.5
    cmap = 'coolwarm'
    if 'SB_high' in save_path:
        use_step  = 59
    else:
        use_step = 99
    
    # print('load distances')
    # load_distances(save_path)
    
    if load_csv:
        print('load simulations')
        
        filter = True
        normalise_population = 10000
        
        if load_csv_path is not None:
            print("Loading CSV files from", load_csv_path)
            read_csvs(load_csv_path, file_name, save_path, save_name=outbreak_file, filter_flag=filter, use_step=use_step, normalise_population=normalise_population)
    
    # print("Loading outbreaks from", outbreak_path)
    # outbreaks = np.load(outbreak_path)['outbreaks']
    
    outbreaks = pd.read_csv('cache/synthetic-graphs/SB_high_v2/sim_beta0.8_gamma0.2_T150_n50000.csv').to_numpy()
    # Plot scatter plot between sim_id 0 and 50000's columns [2:]
    start_ids = [0, 10, 20, 30, 40, 50]
    
    for i in start_ids:
        compare_ids = [i + n for n in [1,2,3,4,5,6,7,8,9]]
        sim_id_0 = outbreaks[outbreaks[:, 0] == i][:50, 2:]
        for n in compare_ids:
            if n in outbreaks[:, 0]:
                sim_id_n = outbreaks[outbreaks[:, 0] == n][:50, 2:]
                
                plt.scatter(sim_id_0.flatten(), sim_id_n.flatten(), alpha=0.5, label = 'n = ' + str(n), s=0.5)
                print('largest difference between sim '+ str(i) + ' and sim ' + str(n) + ': ' + str((sim_id_0 - sim_id_n).max()))
            else: 
                print("Sim ID", n, "not in the outbreaks")
        plt.xlabel('Sim ID Values' + str(i))
        plt.ylabel('Sim ID n Values')
        plt.legend()
        plt.savefig(save_path + 'scatter_sim_' + str(i) + '.png')
        plt.clf()
    
    print("Outbreaks:", outbreaks)
    print("Outbreaks shape:", outbreaks.shape)
    vis_col_distribution(outbreaks, save_path, outbreak_file, use_step, vmin=vmin, vmax=vmax, cmap=cmap)
    
    # plt.hist(outbreaks[:, 1], bins=range(int(outbreaks[:, 1].min()), int(outbreaks[:, 1].max()) + 1), edgecolor='black')
    # plt.xlabel('Time steps')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Time Steps in Outbreaks')
    # plt.savefig(save_path + 'time_distribution.png')
    # plt.clf()
    
def preprocess_SIR_categorical_data(load_csv=False, graph='lattice'):
    save_path = 'cache/outbreaks_' + graph + '/'
    load_csv_path = save_path
    file_name = 'SIR_outbreaks.csv'
    outbreak_file = file_name.split('.')[0] + '.npz'
    vmin = 0
    vmax = 2
    cmap = 'brg'
    
    if load_csv:
        print('load CSVs')
        if 'SB_high' in save_path:
            use_step  = 59
        else:
            use_step = 99
        filter = True
        
        if load_csv_path is not None:
            print("Loading CSV files from", load_csv_path)
            read_csvs(load_csv_path, file_name, save_path, save_name=outbreak_file, filter_flag=filter, use_step=use_step)
        print("Loading outbreaks from", save_path)
    
    # outbreak_path = save_path + outbreak_file
    # outbreaks = np.load(outbreak_path)['outbreaks']
    
    outbreaks = pd.read_csv(save_path + file_name).to_numpy()
    # Plot scatter plot between sim_id 0 and 50000's columns [2:]
    # start_ids = [0, 10, 20, 30, 40, 50]
    start_ids = [0]
    
    for i in start_ids:
        compare_ids = [i + n for n in [1,2,3,4,5,6,7,8,9]]
        sim_id_0 = outbreaks[outbreaks[:, 0] == i][:30, 2:]
        for n in compare_ids:
            sim_id_n = outbreaks[outbreaks[:, 0] == n][:30, 2:]
            plt.scatter(sim_id_0.flatten(), sim_id_n.flatten(), alpha=0.5, label = 'n = ' + str(n), s=0.5)
            print('largest difference between sim '+ str(i) + ' and sim ' + str(n) + ': ' + str((sim_id_0 - sim_id_n).max()))
        plt.xlabel('Sim ID Values' + str(i))
        plt.ylabel('Sim ID n Values')
        plt.legend()
        plt.savefig(save_path + 'scatter_sim_' + str(i) + '.png')
        plt.clf()
    
    print("Outbreaks:", outbreaks)
    print("Outbreaks shape:", outbreaks.shape)
    # vis_col_distribution(outbreaks, save_path, outbreak_file, use_step, vmin=vmin, vmax=vmax, cmap=cmap)
    
def read_us_outbreaks():
    path = 'cache/US_outbreaks/'
    file_name = 'sir_metapop_beta_0.5_gamma_0.2_S_10.csv'
    adj_matrix_name = 'theta_matrix.csv'
    
    outbreaks = pd.read_csv(path + file_name)
    print(outbreaks.head())
    
    adj_matrix = pd.read_csv(path + adj_matrix_name).to_numpy()
    print(adj_matrix.shape)
    

def split_train_test(data_path, train_ratio=0.8):
    # read npz file
    np.random.seed(0)
    data = np.load(data_path)
    outbreaks = data['outbreaks']
    print(outbreaks)
    sim_ids = np.unique(outbreaks[:, 0])
    # select train and test sim ids
    train_sim_ids = np.random.choice(sim_ids, int(len(sim_ids) * train_ratio), replace=False)
    test_sim_ids = sim_ids[~np.isin(sim_ids, train_sim_ids)]
    print("Train sim ids:", train_sim_ids)
    print("Test sim ids:", test_sim_ids)
    print("Train sim ids size:", len(train_sim_ids))
    print("Test sim ids size:", len(test_sim_ids))
    train_outbreaks = outbreaks[np.isin(outbreaks[:, 0], train_sim_ids)]
    test_outbreaks = outbreaks[np.isin(outbreaks[:, 0], test_sim_ids)]
    print("Train outbreaks size:", train_outbreaks.shape)
    print("Test outbreaks size:", test_outbreaks.shape)
    # save train and test outbreaks
    np.savez_compressed(data_path.split('.')[0] + '_train.npz', outbreaks=train_outbreaks)
    np.savez_compressed(data_path.split('.')[0] + '_test.npz', outbreaks=test_outbreaks)
    # return train_outbreaks, test_outbreaks
    print("Train and test outbreaks saved.")

    
if __name__ == "__main__":
    # preprocess_SIR_categorical_data(load_csv=False, graph='lattice')
    # preprocess_SIR_continous_data(load_csv=False, graph='SB_high_v2')
    # read_us_outbreaks()
    split_train_test('cache/outbreaks_lattice/SIR_outbreaks_continuous.npz')