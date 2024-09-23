import os
import networkx as nx
import numpy as np
import random
import pandas as pd
import pickle
from tqdm import tqdm
from typing import Callable, List, Optional
from utils import load_all_graphs
from score_function_sG import TripleScorer

def compute_ATS(
    G: nx.DiGraph,
    G_prime: nx.DiGraph,
    s_G: Callable[[str, str, str], float]
) -> float:
    """
    Compute Aggregated Triple Score (ATS) between two Knowledge Graphs.

    Parameters:
    - G (nx.DiGraph): The raw knowledge graph.
    - G_prime (nx.DiGraph): The perturbed knowledge graph.
    - s_G (Callable): A scoring function that takes (e1, r, e2) and returns a score in [0, 1].

    Returns:
    - float: The ATS score, a value between 0 and 1.
    """
    if len(G_prime.edges) == 0:
        return 0.0  # Avoid division by zero

    scores = []
    for u, v, data in G_prime.edges(data=True):
        r = data.get('relation')
        e1, e2 = u, v
        score = s_G(e1, r, e2)
        scores.append(score)
    
    ats = np.mean(scores)
    return ats

def compute_SC2D(G: nx.DiGraph, G_prime: nx.DiGraph) -> float:
    """
    Compute Similarity in Clustering Coefficient Distribution (SC2D) between two KGs.

    Parameters:
    - G (nx.DiGraph): The raw knowledge graph.
    - G_prime (nx.DiGraph): The perturbed knowledge graph.

    Returns:
    - float: The SC2D score, a value between 0 and 1.
    """
    def get_mean_clustering(graph: nx.DiGraph) -> np.ndarray:
        relations = set(nx.get_edge_attributes(graph, 'relation').values())
        if not relations:
            return np.zeros(len(graph.nodes()))
        
        c_sum = np.zeros(len(graph.nodes()))
        node_list = list(graph.nodes())
        node_index = {node: idx for idx, node in enumerate(node_list)}
        
        for r in relations:
            # Extract subgraph for relation r
            edges_r = [(u, v) for u, v, d in graph.edges(data=True) if d.get('relation') == r]
            subgraph = nx.DiGraph(edges_r)
            undirected_subgraph = subgraph.to_undirected()
            
            # Compute clustering coefficients
            clustering = nx.clustering(undirected_subgraph)
            c_vector = np.zeros(len(node_list))
            for node, c in clustering.items():
                idx = node_index[node]
                c_vector[idx] = c
            c_sum += c_vector
        
        mean_clustering = c_sum / len(relations)
        return mean_clustering

    # Ensure both graphs have the same node ordering
    all_nodes = sorted(set(G.nodes()).union(set(G_prime.nodes())))
    G = G.copy()
    G_prime = G_prime.copy()
    G.add_nodes_from(all_nodes)
    G_prime.add_nodes_from(all_nodes)

    c_o = get_mean_clustering(G)
    c_p = get_mean_clustering(G_prime)
    
    norm_diff = np.linalg.norm(c_o - c_p)
    sc2d = 1 - (norm_diff / (norm_diff + 1))
    return sc2d

def compute_SD2(G: nx.DiGraph, G_prime: nx.DiGraph) -> float:
    """
    Compute Similarity in Degree Distribution (SD2) between two KGs.

    Parameters:
    - G (nx.DiGraph): The raw knowledge graph.
    - G_prime (nx.DiGraph): The perturbed knowledge graph.

    Returns:
    - float: The SD2 score, a value between 0 and 1.
    """
    def get_mean_degree(graph: nx.DiGraph) -> np.ndarray:
        relations = set(nx.get_edge_attributes(graph, 'relation').values())
        if not relations:
            return np.zeros(len(graph.nodes()))
        
        d_sum = np.zeros(len(graph.nodes()))
        node_list = list(graph.nodes())
        node_index = {node: idx for idx, node in enumerate(node_list)}
        
        for r in relations:
            # Extract subgraph for relation r
            edges_r = [(u, v) for u, v, d in graph.edges(data=True) if d.get('relation') == r]
            subgraph = nx.DiGraph(edges_r)
            
            # Compute degrees (in-degree + out-degree)
            degrees = dict(subgraph.degree())
            d_vector = np.zeros(len(node_list))
            for node, deg in degrees.items():
                idx = node_index[node]
                d_vector[idx] = deg
            d_sum += d_vector
        
        mean_degree = d_sum / len(relations)
        return mean_degree

    # Ensure both graphs have the same node ordering
    all_nodes = sorted(set(G.nodes()).union(set(G_prime.nodes())))
    G = G.copy()
    G_prime = G_prime.copy()
    G.add_nodes_from(all_nodes)
    G_prime.add_nodes_from(all_nodes)

    d_o = get_mean_degree(G)
    d_p = get_mean_degree(G_prime)
    
    norm_diff = np.linalg.norm(d_o - d_p)
    sd2 = 1 - (norm_diff / (norm_diff + 1))
    return sd2

def compute_all_metrics(
    G: nx.DiGraph,
    G_prime: nx.DiGraph,
    s_G: Callable[[str, str, str], float]
) -> dict:
    """
    Compute all three metrics: ATS, SC2D, and SD2.

    Parameters:
    - G (nx.DiGraph): The raw knowledge graph.
    - G_prime (nx.DiGraph): The perturbed knowledge graph.
    - s_G (Callable): The ATS scoring function.

    Returns:
    - dict: A dictionary containing the three metrics.
    """
    ats = compute_ATS(G, G_prime, s_G)
    sc2d = compute_SC2D(G, G_prime)
    sd2 = compute_SD2(G, G_prime)
    
    return {
        'ATS': ats,
        'SC2D': sc2d,
        'SD2': sd2
    }
    
    
class GraphPerturber:
    """
    A class to perform various perturbations on a knowledge graph represented as a NetworkX DiGraph.
    
    Supported perturbation methods:
    - Relation Swapping (RS)
    - Relation Replacement (RR)
    - Edge Rewiring (ER)
    - Edge Deletion (ED)
    """
    
    def __init__(self, seed: int = 42, perturbation_level = 0.3):
        """
        Initialize the GraphPerturber with a random seed for reproducibility.
        
        Parameters:
        - seed (int): Random seed.
        """
        self.seed = seed
        random.seed(self.seed)
        self.perturbation_level = perturbation_level
    
    def _perturb_relation_swapping(
        self,
        G: nx.DiGraph,
    ) -> nx.DiGraph:
        """
        Perform Relation Swapping (RS) on the knowledge graph.
        
        Parameters:
        - G (nx.DiGraph): The original knowledge graph.
        - perturbation_level (float): Percentage of edges to perturb (0 < p < 1).
        
        Returns:
        - G_perturbed (nx.DiGraph): The perturbed knowledge graph.
        """
        G_perturbed = G.copy()
        total_edges = G_perturbed.number_of_edges()
        num_swaps = int((self.perturbation_level * total_edges) / 2)
    
        edges = list(G_perturbed.edges(data=True))
        if len(edges) < 2:
            # print("Not enough edges to perform Relation Swapping.")
            return G_perturbed
    
        for _ in range(num_swaps):
            # Randomly select two distinct edges
            edge1, edge2 = random.sample(edges, 2)
            u1, v1, data1 = edge1
            u2, v2, data2 = edge2
    
            # Swap their relations
            G_perturbed[u1][v1]['relation'], G_perturbed[u2][v2]['relation'] = (
                G_perturbed[u2][v2]['relation'],
                G_perturbed[u1][v1]['relation']
            )
    
        # print(f"Relation Swapping: Swapped relations of {num_swaps * 2} edges.")
        return G_perturbed

    def _perturb_relation_replacement(
        self,
        G: nx.DiGraph,
        s_G: Callable[[str, str, str], float],
        relations: List[str]
    ) -> nx.DiGraph:
        """
        Perform Relation Replacement (RR) on the knowledge graph.
        
        Parameters:
        - G (nx.DiGraph): The original knowledge graph.
        - perturbation_level (float): Percentage of edges to perturb (0 < p < 1).
        - s_G (Callable): Scoring function s_G(e1, r, e2).
        - relations (List[str]): List of all possible relations.
        
        Returns:
        - G_perturbed (nx.DiGraph): The perturbed knowledge graph.
        """
        G_perturbed = G.copy()
        total_edges = G_perturbed.number_of_edges()
        num_replacements = int(self.perturbation_level * total_edges)
    
        edges = list(G_perturbed.edges(data=True))
        if not edges:
            # print("No edges available for Relation Replacement.")
            return G_perturbed
    
        for _ in range(num_replacements):
            # Randomly select an edge
            edge = random.choice(edges)
            u, v, data = edge
            original_relation = data.get('relation')
    
            # Find the relation that minimizes the semantic similarity
            # Exclude the original relation to ensure a change
            candidate_relations = [r for r in relations if r != original_relation]
            if not candidate_relations:
                # print("No alternative relations available for replacement.")
                continue
    
            # Compute scores for all candidate relations
            scores = [s_G(u, r, v) for r in candidate_relations]
            if not scores:
                # print("Scoring function returned empty scores.")
                continue
    
            # Select the relation with the minimum score
            min_index = np.argmin(scores)
            new_relation = candidate_relations[min_index]
    
            # Replace the relation
            G_perturbed[u][v]['relation'] = new_relation
    
        # print(f"Relation Replacement: Replaced relations of {num_replacements} edges.")
        return G_perturbed

    def _perturb_edge_rewiring(
        self,
        G: nx.DiGraph,
    ) -> nx.DiGraph:
        """
        Perform Edge Rewiring (ER) on the knowledge graph.

        Edge Rewiring (ER) randomly chooses an edge (e1, r, e2) ∈ T, 
        then replaces e2 with another entity e3 ∈ E \ N1(e1), 
        where N1(e1) represents the 1-hop neighborhood of e1.

        Parameters:
        - G (nx.DiGraph): The original knowledge graph.

        Returns:
        - G_perturbed (nx.DiGraph): The perturbed knowledge graph.
        """
        G_perturbed = G.copy()
        total_edges = G_perturbed.number_of_edges()
        perturbation_level = getattr(self, 'perturbation_level', 0.1)  # Default to 10% if not set
        num_rewirings = int(perturbation_level * total_edges)

        edges = list(G_perturbed.edges(data=True))
        entities = list(G_perturbed.nodes())
        if not edges or not entities:
            # print("No edges or entities available for Edge Rewiring.")
            return G_perturbed

        # Ensure we don't attempt to rewire more edges than available
        num_rewirings = min(num_rewirings, total_edges)

        # Select unique edges to rewire
        edges_to_rewire = random.sample(edges, num_rewirings)

        for edge in edges_to_rewire:
            u, v, data = edge
            original_relation = data.get('relation')

            # Determine 1-hop neighborhood of u
            neighborhood = set(G_perturbed.successors(u)) | set(G_perturbed.predecessors(u))
            neighborhood.add(u)  # Exclude self

            # Possible new entities to connect to
            possible_entities = list(set(entities) - neighborhood)
            if not possible_entities:
                # print(f"No available entities to rewire from entity '{u}'.")
                continue

            # Select a new entity e3
            e3 = random.choice(possible_entities)

            # Check if the new edge already exists
            if G_perturbed.has_edge(u, e3):
                # print(f"Edge ({u}, {e3}) already exists. Skipping rewiring for this edge.")
                continue

            # Replace the tail entity
            try:
                G_perturbed.remove_edge(u, v)
                G_perturbed.add_edge(u, e3, relation=original_relation)
                # print(f"Rewired edge ({u}, {v}) to ({u}, {e3}) with relation '{original_relation}'.")
            except nx.NetworkXError as e:
                # print(f"Error rewiring edge ({u}, {v}): {e}")
                continue

        # print(f"Edge Rewiring: Rewired {num_rewirings} edges out of {total_edges}.")
        return G_perturbed

    def _perturb_edge_deletion(
        self,
        G: nx.DiGraph,
    ) -> nx.DiGraph:
        """
        Perform Edge Deletion (ED) on the knowledge graph.
        
        Parameters:
        - G (nx.DiGraph): The original knowledge graph.
        - perturbation_level (float): Percentage of edges to perturb (0 < p < 1).
        
        Returns:
        - G_perturbed (nx.DiGraph): The perturbed knowledge graph.
        """
        G_perturbed = G.copy()
        total_edges = G_perturbed.number_of_edges()
        num_deletions = int(self.perturbation_level * total_edges)
    
        edges = list(G_perturbed.edges())
        if not edges:
            # print("No edges available for Edge Deletion.")
            return G_perturbed
    
        edges_to_delete = random.sample(edges, min(num_deletions, len(edges)))
    
        G_perturbed.remove_edges_from(edges_to_delete)
    
        # print(f"Edge Deletion: Deleted {len(edges_to_delete)} edges.")
        return G_perturbed

    def apply_perturbation(
        self,
        G: nx.DiGraph,
        method: str,
        s_G: Optional[Callable[[str, str, str], float]] = None,
        relations: Optional[List[str]] = None
    ) -> nx.DiGraph:
        """
        Apply a specified perturbation method to the knowledge graph.
        
        Parameters:
        - G (nx.DiGraph): The original knowledge graph.
        - method (str): Perturbation method to apply ('RS', 'RR', 'ER', 'ED').
        - perturbation_level (float): Percentage of edges to perturb (0 < p < 1).
        - s_G (Callable, optional): Scoring function for RR. Required if method is 'RR'.
        - relations (List[str], optional): List of all possible relations. Required if method is 'RR'.
        
        Returns:
        - G_perturbed (nx.DiGraph): The perturbed knowledge graph.
        
        Raises:
        - ValueError: If perturbation_level is not in (0,1], or if required parameters for 'RR' are missing.
        """
        method = method.upper()
        if not (0 < self.perturbation_level <= 1):
            raise ValueError("Perturbation level must be between 0 and 1.")
    
        if method == 'RS':
            if G.number_of_edges() < 2:
                # print("Not enough edges to perform Relation Swapping.")
                return G.copy()
            return self._perturb_relation_swapping(G)
    
        elif method == 'RR':
            if s_G is None or relations is None:
                raise ValueError("Scoring function 's_G' and 'relations' list must be provided for Relation Replacement.")
            if G.number_of_edges() == 0:
                # print("No edges available for Relation Replacement.")
                return G.copy()
            return self._perturb_relation_replacement(G, s_G, relations)
    
        elif method == 'ER':
            if G.number_of_edges() == 0 or G.number_of_nodes() < 2:
                # print("Not enough edges or entities to perform Edge Rewiring.")
                return G.copy()
            return self._perturb_edge_rewiring(G)
    
        elif method == 'ED':
            if G.number_of_edges() == 0:
                # print("No edges available for Edge Deletion.")
                return G.copy()
            return self._perturb_edge_deletion(G)
    
        else:
            raise ValueError(f"Unknown perturbation method '{method}'. Choose from 'RS', 'RR', 'ER', 'ED'.")
        
        
if __name__ == "__main__":
    # ----------------------------
    # Load data and preprocess
    # ----------------------------
    pruned_ppr_graphs = load_all_graphs("subgraphs/pruned_ppr/", sample_size=None)
    G_all = nx.compose_all(pruned_ppr_graphs)
    
    # ----------------------------
    # Initialize the Scoring Function
    # ----------------------------
    print("Initializing the scoring function...")
    scorer = TripleScorer(model_path="/data/yuansui/link_prediction_model")
    all_relations = list(set(nx.get_edge_attributes(G_all, 'relation').values()))
    print(f"All Relations: {all_relations}")
    
    # ----------------------------
    # Initialize the Metrics Dictionary
    # ----------------------------
    perturbation_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    try:
        os.mkdir("perturbed_graphs")
    except FileExistsError:
        pass
    save_directory = "perturbed_graphs"
    
    comprehensive_metrics = {
        'Perturbation_Level': [],
        'Method': [],
        'ATS': [],
        'SC2D': [],
        'SD2': []
    }
    
    # ----------------------------
    # Apply Perturbation Methods Across Different Levels
    # ----------------------------
    for perturbation_level in perturbation_levels:
        print(f"\n===== Perturbation Level: {perturbation_level} =====")
        graph_perturber = GraphPerturber(seed=42, perturbation_level=perturbation_level)
        
        # Initialize dictionaries to store perturbed graphs per method
        perturbed_graphs = {
            'RS': [],
            'RR': [],
            'ER': [],
            'ED': []
        }
        
        # Initialize a temporary metrics dictionary for the current perturbation level
        metrics_dict = {
            'Method': [],
            'ATS': [],
            'SC2D': [],
            'SD2': []
        }
    
        # ----------------------------
        # Apply Perturbation Methods
        # ----------------------------
        for G in tqdm(pruned_ppr_graphs, desc=f"Perturbing Graphs at Level {perturbation_level}", leave=True):
                        
            # 1. Relation Swapping (RS)
            # print("\n--- Applying Relation Swapping (RS) ---")
            G_RS = graph_perturber.apply_perturbation(
                G=G,
                method='RS',
            )
            perturbed_graphs['RS'].append(G_RS)
            metrics_RS = compute_all_metrics(G, G_RS, scorer.score)
            metrics_dict['Method'].append('RS')
            metrics_dict['ATS'].append(metrics_RS['ATS'])
            metrics_dict['SC2D'].append(metrics_RS['SC2D'])
            metrics_dict['SD2'].append(metrics_RS['SD2'])

            # 2. Relation Replacement (RR)
            # print("\n--- Applying Relation Replacement (RR) ---")
            G_RR = graph_perturber.apply_perturbation(
                G=G,
                method='RR',
                s_G=scorer.score,
                relations=all_relations,
            )
            perturbed_graphs['RR'].append(G_RR)
            metrics_RR = compute_all_metrics(G, G_RR, scorer.score)
            metrics_dict['Method'].append('RR')
            metrics_dict['ATS'].append(metrics_RR['ATS'])
            metrics_dict['SC2D'].append(metrics_RR['SC2D'])
            metrics_dict['SD2'].append(metrics_RR['SD2'])

            # 3. Edge Rewiring (ER)
            # print("\n--- Applying Edge Rewiring (ER) ---")
            G_ER = graph_perturber.apply_perturbation(
                G=G,
                method='ER',
            )
            perturbed_graphs['ER'].append(G_ER)
            metrics_ER = compute_all_metrics(G, G_ER, scorer.score)
            metrics_dict['Method'].append('ER')
            metrics_dict['ATS'].append(metrics_ER['ATS'])
            metrics_dict['SC2D'].append(metrics_ER['SC2D'])
            metrics_dict['SD2'].append(metrics_ER['SD2'])

            # 4. Edge Deletion (ED)
            # print("\n--- Applying Edge Deletion (ED) ---")
            G_ED = graph_perturber.apply_perturbation(
                G=G,
                method='ED',
            )
            perturbed_graphs['ED'].append(G_ED)
            metrics_ED = compute_all_metrics(G, G_ED, scorer.score)
            metrics_dict['Method'].append('ED')
            metrics_dict['ATS'].append(metrics_ED['ATS'])
            metrics_dict['SC2D'].append(metrics_ED['SC2D'])
            metrics_dict['SD2'].append(metrics_ED['SD2'])
    
        # ----------------------------
        # Save Perturbed Graphs
        # ----------------------------
        pkl_filename = f"perturbed_graphs_level_{int(perturbation_level * 100)}.pkl"
        pkl_path = os.path.join(save_directory, pkl_filename)
        with open(pkl_path, "wb") as f:
            pickle.dump(perturbed_graphs, f)
        print(f"Saved perturbed graphs for perturbation_level {perturbation_level} to '{pkl_path}'")
        
        # ----------------------------
        # Aggregate Metrics
        # ----------------------------
        metrics_df = pd.DataFrame(metrics_dict)
        metrics_df['Perturbation_Level'] = perturbation_level  # Add perturbation level info
        comprehensive_metrics['Perturbation_Level'].extend(metrics_df['Perturbation_Level'].tolist())
        comprehensive_metrics['Method'].extend(metrics_df['Method'].tolist())
        comprehensive_metrics['ATS'].extend(metrics_df['ATS'].tolist())
        comprehensive_metrics['SC2D'].extend(metrics_df['SC2D'].tolist())
        comprehensive_metrics['SD2'].extend(metrics_df['SD2'].tolist())
        
        # Compute and display average metrics for the current perturbation level
        average_metrics = metrics_df.groupby('Method').mean().reset_index()
        print("\n===== Average Metrics for Current Perturbation Level =====")
        print(average_metrics)
    
    # ----------------------------
    # Save Comprehensive Metrics
    # ----------------------------
    comprehensive_metrics_df = pd.DataFrame(comprehensive_metrics)
    metrics_save_path = "comprehensive_perturb_metrics.csv"
    
    grouping_columns = ['Method', 'Perturbation_Level']
    average_metrics = comprehensive_metrics_df.groupby(grouping_columns).mean().reset_index()
    average_metrics.to_csv(f"{save_directory}/metrics_save_path", index=False)
    print(f"\nSaved comprehensive metrics to '{metrics_save_path}'")
    
    # Optionally, display the entire metrics dataframe
    print("\n===== Comprehensive Metrics =====")
    print(average_metrics)