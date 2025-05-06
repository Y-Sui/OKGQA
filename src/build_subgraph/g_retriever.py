import networkx as nx
import numpy as np
import pandas as pd
import heapq
import os
import heapq
from openai import OpenAI
from typing import List, Tuple, Dict
from pcst_fast import pcst_fast
from preprocess import preprocess_graph
from src.utils import load_all_graphs


def triplet_retrieval(graph, k):
    """
    Retrieve triplets (edges) based on the combined similarity scores of nodes and edges.
    """
    # Compute combined similarity for each edge (triplet)
    triplet_similarities = []
    for u, v, data in graph.edges(data=True):
        node_u_sim = graph.nodes[u].get('similarity', 0)
        node_v_sim = graph.nodes[v].get('similarity', 0)
        edge_sim = data.get('similarity', 0)
        # Combine the similarities (you can adjust the weights as needed)
        combined_sim = node_u_sim + node_v_sim + edge_sim
        triplet_similarities.append(((u, v), combined_sim))
    
    # Sort triplets based on the combined similarity
    triplet_similarities.sort(key=lambda x: x[1], reverse=True)
    # Select the top-k triplets
    top_k_triplets = triplet_similarities[:k]
    # Retrieve the triplets with their data
    triplets = []
    for (u, v), _ in top_k_triplets:
        triplets.append((u, graph.edges[u, v]['relation'], v))
    return triplets

def path_retrieval(graph, max_path_length=8, max_paths=5):
    """
    Retrieve paths based on the prize assignment and cost allocation.
    Returns a list of paths, where each path is a list of alternating nodes and relations.
    """
    # Start from high-prize nodes
    node_prizes = [(node, data.get('prize', 0)) for node, data in graph.nodes(data=True) if data.get('prize', 0) > 0]
    node_prizes.sort(key=lambda x: x[1], reverse=True)
    
    paths = []
    pq = []
    # Initialize priority queue with paths starting from high-prize nodes
    for node, prize in node_prizes:
        path = [node]  # Start with the node
        score = prize  # Initial score is the node's prize
        heapq.heappush(pq, (-score, path))  # Negative score for max-heap behavior
    
    visited_paths = set()
    while pq and len(paths) < max_paths:
        neg_score, path = heapq.heappop(pq)
        current_node = path[-1]  # The last node in the path
        path_nodes = tuple(n for n in path if isinstance(n, str) or isinstance(n, int))  # Node identifiers
        
        if path_nodes in visited_paths:
            continue
        visited_paths.add(path_nodes)
        
        # Check path length (number of nodes)
        num_nodes = (len(path) + 1) // 2  # Since path is nodes and relations alternating
        if num_nodes > max_path_length:
            continue
        
        # Save the path
        paths.append((-neg_score, path.copy()))
        
        # Extend the path if it hasn't reached max_path_length
        if num_nodes < max_path_length:
            for neighbor in graph.successors(current_node):
                edge_data = graph.get_edge_data(current_node, neighbor)
                relation = edge_data.get('relation')
                edge_prize = edge_data.get('prize', 0)
                edge_cost = edge_data.get('cost', 1.0)
                node_data = graph.nodes[neighbor]
                neighbor_prize = node_data.get('prize', 0)
                
                if neighbor in path_nodes:
                    continue  # Avoid cycles
                
                # Calculate new score
                new_score = -neg_score + neighbor_prize + edge_prize - edge_cost
                
                # Create new path
                new_path = path + [relation, neighbor]
                heapq.heappush(pq, (-new_score, new_path))
    
    # Sort the found paths based on score in descending order
    paths.sort(key=lambda x: x[0], reverse=True)
    
    return paths

def subgraph_retrieval(graph):
    """
    Retrieve a subgraph using the Prize-Collecting Steiner Tree (PCST) algorithm.
    """

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return nx.DiGraph()

    # Map nodes to indices
    node_id_map = {node: idx for idx, node in enumerate(graph.nodes())}
    num_nodes = len(node_id_map)
    prizes = np.zeros(num_nodes, dtype=float)

    # Collect node prizes
    for node, idx in node_id_map.items():
        prizes[idx] = graph.nodes[node].get('prize', 0.0)

    edges_list = []
    costs = []
    virtual_node_map = {}
    edge_id_map = {}
    node_idx = num_nodes  # Index for virtual nodes

    for u, v, data in graph.edges(data=True):
        u_idx = node_id_map[u]
        v_idx = node_id_map[v]
        edge_prize = data.get('prize', 0.0)
        edge_cost = data.get('cost', 1.0)

        if edge_prize <= edge_cost:
            cost = edge_cost - edge_prize
            edges_list.append((u_idx, v_idx))
            costs.append(cost)
            edge_id_map[(u_idx, v_idx)] = (u, v)
        else:
            # Introduce a virtual node
            virtual_node_idx = node_idx
            node_idx += 1
            virtual_node_map[virtual_node_idx] = (u, v)
            virtual_node_prize = edge_prize - edge_cost
            prizes = np.append(prizes, virtual_node_prize)
            # Add edges connected to the virtual node
            edges_list.append((u_idx, virtual_node_idx))
            costs.append(0.0)
            edges_list.append((virtual_node_idx, v_idx))
            costs.append(0.0)
            # No need to map these edges, as they are virtual

    # Prepare edges and costs for PCST
    edges_array = np.array(edges_list, dtype=int)
    costs_array = np.array(costs, dtype=float)

    # Run PCST
    root = -1  # Unrooted PCST
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0

    vertices, edges_selected = pcst_fast(edges_array, prizes, costs_array, root, num_clusters, pruning, verbosity_level)

    # Map back the selected nodes and edges
    selected_nodes = set()
    selected_edges = set()

    for node_idx in vertices:
        if node_idx < num_nodes:
            node = list(node_id_map.keys())[list(node_id_map.values()).index(node_idx)]
            selected_nodes.add(node)
        else:
            # Virtual node, get the corresponding edge
            edge = virtual_node_map.get(node_idx)
            if edge:
                selected_edges.add(edge)
                selected_nodes.update(edge)

    for edge_idx in edges_selected:
        u_idx, v_idx = edges_array[edge_idx]
        if u_idx < num_nodes and v_idx < num_nodes:
            # Original edge
            edge = edge_id_map.get((u_idx, v_idx))
            if edge:
                selected_edges.add(edge)
                selected_nodes.update(edge)
        elif u_idx < num_nodes and v_idx >= num_nodes:
            # Edge from node to virtual node
            virtual_node_idx = v_idx
            edge = virtual_node_map.get(virtual_node_idx)
            if edge:
                selected_edges.add(edge)
                selected_nodes.update(edge)
        elif u_idx >= num_nodes and v_idx < num_nodes:
            # Edge from virtual node to node
            virtual_node_idx = u_idx
            edge = virtual_node_map.get(virtual_node_idx)
            if edge:
                selected_edges.add(edge)
                selected_nodes.update(edge)

    # Create the subgraph
    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(selected_nodes)

    for u, v in selected_edges:
        subgraph.add_edge(u, v, **graph.edges[u, v])
        
    textual_subgraphs = []
    for edge in subgraph.edges(data=True):
        textual_subgraphs.append((edge[0], edge[2]['relation'], edge[1]))

    return subgraph, textual_subgraphs


def main():
    df = pd.read_csv("query/filtered_questions_63a0f8a06513_valid.csv")
    pruned_ppr_graphs = load_all_graphs("subgraphs/pruned_ppr/")
    graphs = [item['graph'] for item in pruned_ppr_graphs][:2]
    queries = df["question"][:2]
    
    f = open("output.txt", "w")

    for i, (query, G) in enumerate(zip(queries, graphs)):
        preprocess_graph(G=G, query_text=query, embedding_model="sbert", top_k_nodes=10, top_k_edges=10)
        
        # Triplet retrieval
        triplets = triplet_retrieval(G, k=10)
        print("Triplet Retrieval:")
        for triplet in triplets:
            print(triplet)

        # Path retrieval
        paths = path_retrieval(G, max_path_length=6, max_paths=5)
        print("\nPath Retrieval:")
        for score, path in paths:
            print(f"Score: {score}, Path: {path}")

        # Subgraph retrieval
        subgraph, textual_subgraph = subgraph_retrieval(G)
        print("\nSubgraph Retrieval:")
        print(textual_subgraph)
        
        f.write(f"Query: {query}\n")
        f.write(f"Triplet Retrieval: {textual_subgraph}\n")
        
if __name__ == "__main__":
    main()