import networkx as nx
import numpy as np
import heapq
import os
import pulp
from openai import OpenAI
from typing import List, Tuple, Dict


def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    Retrieve the embedding for a given text using OpenAI's embedding API.

    Parameters:
    - text (str): The input text to embed.
    - model (str): The embedding model to use.

    Returns:
    - List[float]: The embedding vector.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = client.embeddings.create(input=text, model=model)
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error obtaining embedding for text: {text}\n{e}")
        return []

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute the cosine similarity between two vectors.

    Parameters:
    - vec1 (List[float]): First vector.
    - vec2 (List[float]): Second vector.

    Returns:
    - float: Cosine similarity.
    """
    a = np.array(vec1)
    b = np.array(vec2)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_node_prizes(query_embedding: List[float], G: nx.Graph, top_k: int) -> dict:
    """
    Assign prizes to nodes based on their relevance to the query using embeddings.

    Parameters:
    - query_embedding (List[float]): Embedding of the query.
    - G (networkx.Graph): The input graph with node attributes.
    - top_k (int): Number of top nodes to assign prizes.

    Returns:
    - dict: A dictionary mapping nodes to their prizes.
    """
    node_prizes = {}
    similarities = {}
    
    # Compute similarity for each node
    for node in G.nodes():
        node_text = G.nodes[node].get('text', '')
        node_embedding = G.nodes[node].get('embedding', [])
        if not node_embedding:
            node_embedding = get_embedding(node_text)
            G.nodes[node]['embedding'] = node_embedding  # Cache embedding
        sim = cosine_similarity(query_embedding, node_embedding)
        similarities[node] = sim
    
    # Rank nodes based on similarity
    ranked_nodes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Assign prizes: top_k nodes get prizes from top_k to 1
    for i, (node, sim) in enumerate(ranked_nodes[:top_k]):
        node_prizes[node] = top_k - i  # Prize decreases with rank
    
    return node_prizes

def compute_edge_prizes(query_embedding: List[float], G: nx.Graph, top_k: int) -> dict:
    """
    Assign prizes to edges based on their relevance to the query using embeddings.

    Parameters:
    - query_embedding (List[float]): Embedding of the query.
    - G (networkx.Graph): The input graph with edge attributes.
    - top_k (int): Number of top edges to assign prizes.

    Returns:
    - dict: A dictionary mapping edges to their prizes.
    """
    edge_prizes = {}
    similarities = {}
    
    # Compute similarity for each edge
    for edge in G.edges():
        edge_text = G.edges[edge].get('text', '')
        edge_embedding = G.edges[edge].get('embedding', [])
        if not edge_embedding:
            edge_embedding = get_embedding(edge_text)
            G.edges[edge]['embedding'] = edge_embedding  # Cache embedding
        sim = cosine_similarity(query_embedding, edge_embedding)
        similarities[edge] = sim
    
    # Rank edges based on similarity
    ranked_edges = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Assign prizes: top_k edges get prizes from top_k to 1
    for i, (edge, sim) in enumerate(ranked_edges[:top_k]):
        edge_prizes[edge] = top_k - i  # Prize decreases with rank
    
    return edge_prizes

def assign_edge_costs(G: nx.Graph, default_cost: float = 1.0) -> nx.Graph:
    """
    Assign costs to edges in the graph.

    Parameters:
    - G (networkx.Graph): The input graph.
    - default_cost (float): The default cost to assign to each edge.

    Returns:
    - networkx.Graph: The graph with updated edge costs.
    """
    for edge in G.edges():
        if 'cost' not in G.edges[edge]:
            G.edges[edge]['cost'] = default_cost
    return G

def find_reasoning_paths(
    G: nx.Graph,
    node_prizes: dict,
    edge_prizes: dict,
    C_e: float,
    max_paths: int = 5,
    max_length: int = 10
) -> List[Tuple[List, float]]:
    """
    Retrieve reasoning paths from the graph based on node and edge prizes.

    Parameters:
    - G (networkx.Graph): The input graph.
    - node_prizes (dict): Prizes assigned to nodes.
    - edge_prizes (dict): Prizes assigned to edges.
    - C_e (float): Cost assigned to each edge.
    - max_paths (int): Maximum number of paths to retrieve.
    - max_length (int): Maximum length of each path.

    Returns:
    - list: A list of tuples, each containing a path (list of nodes and relations) and its score.
    """
    # Identify starting nodes (nodes with non-zero prizes)
    starting_nodes = [n for n, p in node_prizes.items() if p > 0]
    paths = []
    visited_paths = set()

    for start in starting_nodes:
        # Priority queue: (negative_score, path)
        heap = []
        initial_score = node_prizes.get(start, 0)
        heapq.heappush(heap, (-(initial_score), [start]))
        
        while heap and len(paths) < max_paths:
            neg_score, path = heapq.heappop(heap)
            current_score = -neg_score
            current_node = path[-1]
            
            if len(path) >= max_length:
                continue  # Exceeds maximum path length
            
            # Explore neighbors
            for neighbor in G.neighbors(current_node):
                if neighbor in path:
                    continue  # Avoid cycles
                
                edge = (current_node, neighbor) if (current_node, neighbor) in G.edges() else (neighbor, current_node)
                edge_prize = edge_prizes.get(edge, 0)
                node_prize = node_prizes.get(neighbor, 0)
                edge_cost = G.edges[edge].get('cost', C_e)
                
                # Calculate new score
                new_score = current_score + edge_prize + node_prize - edge_cost
                new_path = path + [edge, neighbor]  # Include edge in the path
                
                # Avoid revisiting the same path
                path_tuple = tuple(new_path)
                if path_tuple in visited_paths:
                    continue
                visited_paths.add(path_tuple)
                
                # Push the new path to the heap
                heapq.heappush(heap, (-(new_score), new_path))
                
                # Add to the final paths list
                paths.append((new_path, new_score))
                
                if len(paths) >= max_paths:
                    break

    # Sort paths based on their scores in descending order
    sorted_paths = sorted(paths, key=lambda x: x[1], reverse=True)
    
    # Extract only the paths (without scores)
    final_paths = [path for path, score in sorted_paths[:max_paths]]
    return final_paths

def retrieve_reasoning_paths(
    query: str,
    G: nx.Graph,
    top_k_nodes: int = 10,
    top_k_edges: int = 10,
    edge_cost: float = 1.0,
    max_paths: int = 5,
    max_length: int = 10
) -> List[List]:
    """
    Main function to retrieve reasoning paths based on a query and graph using OpenAI embeddings.

    Parameters:
    - query (str): The input query.
    - G (networkx.Graph): The input graph.
    - top_k_nodes (int): Number of top nodes to consider for prizes.
    - top_k_edges (int): Number of top edges to consider for prizes.
    - edge_cost (float): Default cost assigned to each edge.
    - max_paths (int): Maximum number of paths to retrieve.
    - max_length (int): Maximum length of each path.

    Returns:
    - list: A list of reasoning paths, each including nodes and their relations.
    """
    # Step 1: Generate embedding for the query
    query_embedding = get_embedding(query)
    if not query_embedding:
        print("Failed to obtain query embedding.")
        return []

    # Step 2: Assign prizes to nodes
    node_prizes = compute_node_prizes(query_embedding, G, top_k_nodes)

    # Step 3: Assign prizes to edges
    edge_prizes = compute_edge_prizes(query_embedding, G, top_k_edges)

    # Step 4: Assign costs to edges
    G = assign_edge_costs(G, default_cost=edge_cost)

    # Step 5: Retrieve reasoning paths
    paths = find_reasoning_paths(
        G=G,
        node_prizes=node_prizes,
        edge_prizes=edge_prizes,
        C_e=edge_cost,
        max_paths=max_paths,
        max_length=max_length
    )

    return paths

# ---------------------- Usage Example ----------------------

if __name__ == "__main__":
    # Create a sample graph
    G = nx.Graph()

    # Add nodes with 'text' attributes
    G.add_node(1, text="Climate Change")
    G.add_node(2, text="Polar Bears")
    G.add_node(3, text="Sea Ice")
    G.add_node(4, text="Food Sources")
    G.add_node(5, text="Migration Patterns")
    G.add_node(6, text="Ecosystem Balance")

    # Add edges with 'text' attributes
    G.add_edge(1, 3, text="Affects")
    G.add_edge(3, 2, text="Habitat")
    G.add_edge(1, 4, text="Impacts")
    G.add_edge(4, 2, text="Availability")
    G.add_edge(2, 5, text="Influences")
    G.add_edge(5, 6, text="Maintains")
    G.add_edge(6, 1, text="Regulates")

    # Define the query
    query = "Explain the impact of climate change on polar bear populations."

    # Retrieve reasoning paths
    reasoning_paths = retrieve_reasoning_paths(
        query=query,
        G=G,
        top_k_nodes=5,
        top_k_edges=5,
        edge_cost=1.0,
        max_paths=5,
        max_length=10
    )

    # Display the reasoning paths with relations
    print("Retrieved Reasoning Paths:")
    for idx, path in enumerate(reasoning_paths, 1):
        # Initialize an empty string for the formatted path
        path_text = ""
        # Iterate through the path list, which contains nodes and edges alternately
        for i in range(len(path)):
            if i % 2 == 0:
                # Node
                node_id = path[i]
                node_text = G.nodes[node_id]['text']
                path_text += node_text
            else:
                # Edge
                edge = path[i]
                relation = G.edges[edge]['text']
                path_text += f" --{relation}--> "
        print(f"Path {idx}: {path_text}")