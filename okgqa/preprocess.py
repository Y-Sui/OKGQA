import os
import networkx as nx
import numpy as np
import pickle
import pandas as pd
from openai import OpenAI
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv
from utils import load_all_graphs
from sentence_transformers import SentenceTransformer

load_dotenv()

def get_embedding(text, model: str = "text-embedding-ada-002") -> List[float]:
    """
    Retrieve the embedding for a given text using OpenAI's embedding API.

    Parameters:
    - text (str): The input text to embed.
    - model (str): The embedding model to use.

    Returns:
    - List[float]: The embedding vector.
    """
    if model == "text-embedding-ada-002":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        embeddings = []
        try:
            response = client.embeddings.create(input=text, model=model)
            for data in response.data:
                embeddings.append(data.embedding)
            
            return embeddings
        except Exception as e:
            print(f"Error obtaining embedding for text: {text}\n{e}")
            return []
        
    elif model == "sbert":
        model = SentenceTransformer('all-MiniLM-L6-v2')
        try:
            embeddings = model.encode(text, convert_to_numpy=True).tolist()
            return embeddings
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

def compute_node_prizes(query_embedding: List[float], G: nx.Graph, top_k: int) -> Dict:
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
    nodes_to_embed = []
    node_ids = []
    
    # Compute similarity for each node
    for node in G.nodes():
        node_text = node
        node_embedding = G.nodes[node].get('embedding', [])
        if not node_embedding:
            nodes_to_embed.append(node_text)
            node_ids.append(node)
            
    # Generate embeddings in batches
    if nodes_to_embed:
        print(f"Generating embeddings for {len(nodes_to_embed)} nodes in batches...")
        node_embeddings = get_embedding(nodes_to_embed)
        for node, embedding in zip(node_ids, node_embeddings):
            if embedding:
                G.nodes[node]['embedding'] = embedding  # Cache embedding
            else:
                print(f"Failed to generate embedding for node {node}.")
    
    # Compute similarities
    for node in G.nodes():
        node_embedding = G.nodes[node].get('embedding', [])
        if not node_embedding:
            similarities[node] = 0.0  # Assign zero similarity if embedding is missing
            continue
        sim = cosine_similarity(query_embedding, node_embedding)
        similarities[node] = sim
    
    # Rank nodes based on similarity
    ranked_nodes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Assign prizes: top_k nodes get prizes from top_k to 1
    for i, (node, sim) in enumerate(ranked_nodes[:top_k]):
        node_prizes[node] = top_k - i  # Prize decreases with rank
        G.nodes[node]['similarity'] = sim  # Store similarity
        G.nodes[node]['prize'] = top_k - i  # Store prize
    
    return node_prizes

def compute_edge_prizes(query_embedding: List[float], G: nx.Graph, top_k: int) -> Dict:
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
    edges_to_embed = []
    edge_ids = []

    # Collect edges that need embeddings
    for edge in G.edges():
        edge_text = G.edges[edge].get('relation', '')
        edge_embedding = G.edges[edge].get('embedding', [])
        if not edge_embedding and edge_text:
            edges_to_embed.append(edge_text)
            edge_ids.append(edge)
    
    # Generate embeddings in batches
    if edges_to_embed:
        print(f"Generating embeddings for {len(edges_to_embed)} edges in batches...")
        edge_embeddings = get_embedding(edges_to_embed)
        for edge, embedding in zip(edge_ids, edge_embeddings):
            if embedding:
                G.edges[edge]['embedding'] = embedding  # Cache embedding
            else:
                print(f"Failed to generate embedding for edge {edge}.")
    
    # Compute similarities
    for edge in G.edges():
        edge_embedding = G.edges[edge].get('embedding', [])
        if not edge_embedding:
            similarities[edge] = 0.0  # Assign zero similarity if embedding is missing
            continue
        sim = cosine_similarity(query_embedding, edge_embedding)
        similarities[edge] = sim
    
    # Rank edges based on similarity
    ranked_edges = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Assign prizes: top_k edges get prizes from top_k to 1
    for i, (edge, sim) in enumerate(ranked_edges[:top_k]):
        edge_prizes[edge] = top_k - i  # Prize decreases with rank
        G.edges[edge]['similarity'] = sim  # Store similarity
        G.edges[edge]['prize'] = top_k - i  # Store prize
    
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

def preprocess_graph(
    G: nx.Graph,
    query_text: str,
    top_k_nodes: int,
    top_k_edges: int,
    output_pkl_path: str,
    default_edge_cost: float = 1.0,
    embedding_model: str = "text-embedding-ada-002"
) -> None:
    """
    Preprocess the graph by generating embeddings for each node and edge,
    computing similarities with the query, assigning prizes, and saving the graph.

    Parameters:
    - G (nx.Graph): The input NetworkX graph.
    - query_text (str): The query text for which embeddings and similarities are computed.
    - top_k_nodes (int): Number of top nodes to assign prizes.
    - top_k_edges (int): Number of top edges to assign prizes.
    - output_pkl_path (str): File path to save the processed graph as a pickle file.
    - default_edge_cost (float, optional): Default cost to assign to edges if not present. Defaults to 1.0.
    - embedding_model (str, optional): The OpenAI embedding model to use. Defaults to "text-embedding-ada-002".

    Returns:
    - None
    """
    # Assign default costs to edges if not already set
    assign_edge_costs(G, default_cost=default_edge_cost)

    # Generate embeddings for all nodes
    for node in G.nodes():
        node_text = node
        if not node_text:
            continue
        if 'embedding' not in G.nodes[node] or not G.nodes[node]['embedding']:
            embedding = get_embedding(node_text, model=embedding_model)
            if embedding:
                G.nodes[node]['embedding'] = embedding

    # Generate embeddings for all edges
    # print("Generating embeddings for edges...")
    for edge in G.edges():
        edge_text = G.edges[edge].get('relation', '')
        if not edge_text:
            continue
        if 'embedding' not in G.edges[edge] or not G.edges[edge]['embedding']:
            embedding = get_embedding(edge_text, model=embedding_model)
            if embedding:
                G.edges[edge]['embedding'] = embedding

    # Compute query embedding
    query_embedding = get_embedding(query_text, model=embedding_model)
    if not query_embedding:
        return
    G.graph['query_embedding'] = query_embedding

    # Compute and assign prizes to nodes
    node_prizes = compute_node_prizes(query_embedding, G, top_k_nodes)

    # Compute and assign prizes to edges
    edge_prizes = compute_edge_prizes(query_embedding, G, top_k_edges)


if __name__ == "__main__":
    
    # text = ["The quick brown fox jumps over the lazy dog.", "asd"]
    # # embeddigs = get_embedding(text, model="sbert")
    # embeddigs = get_embedding(text, model="text-embedding-ada-002")
    
    # print(embeddigs)
    
    pruned_ppr_graphs = load_all_graphs("subgraphs/pruned_ppr/", sample_size=5)
    
    graph = pruned_ppr_graphs[0]
    # for node in graph.nodes():
    #     print("node ",node)
        
    # for edge in graph.edges():
    #     print("edge ",edge)
    #     print("relation ",graph.edges[edge]['relation'])
        
    query = "Find the most relevant nodes and edges related to node two."
        
    preprocess_graph(
        G=graph,
        query_text=query,
        top_k_nodes=5,
        top_k_edges=5,
        output_pkl_path="processed_graph.pkl",
        default_edge_cost=1.5,  # Optional: specify default edge cost
        embedding_model="text-embedding-ada-002"  # Optional: specify embedding model
    )
    
    # for node in graph.nodes(data=True):
    #     node_id = node[0]
    #     attributes = node[1]
    #     prize = attributes.get('prize', 'no prize')
    #     similarity = attributes.get('similarity', 'No Similarity')
    #     print(f"Node {node_id}: Prize = {prize}, Similarity = {similarity}")
        
    # for edge in graph.edges(data=True):
    #     edge_id = edge[:2]
    #     attributes = edge[2]
    #     prize = attributes.get('prize', 'no prize')
    #     similarity = attributes.get('similarity', 'No Similarity')
    #     print(f"Edge {edge_id}: Prize = {prize}, Similarity = {similarity}, Edge Cost = {attributes.get('cost', 'No Cost')}, Edge Text = {attributes.get('relation', 'No Text')}")
    
    # # Example: Creating a simple graph
    # G = nx.Graph()

    # # Adding nodes with 'text' attributes
    # G.add_node(1, text="Node one description.")
    # G.add_node(2, text="Node two description.")
    # G.add_node(3, text="Node three description.")

    # # Adding edges with 'text' attributes
    # G.add_edge(1, 2, relation="Edge between node one and two.")
    # G.add_edge(2, 3, relation="Edge between node two and three.")
    # G.add_edge(1, 3, relation="Edge between node one and three.")

    # # Define query
    # query = "Find the most relevant nodes and edges related to node two."

    # preprocess_graph(
    #     G=G,
    #     query_text=query,
    #     top_k_nodes=5,
    #     top_k_edges=5,
    #     output_pkl_path="processed_graph.pkl",
    #     default_edge_cost=1.5,  # Optional: specify default edge cost
    #     embedding_model="text-embedding-ada-002"  # Optional: specify embedding model
    # )
    
    # print("Preprocessing completed.")
    # for node in G.nodes(data=True):
    #     node_id = node[0]
    #     attributes = node[1]
    #     prize = attributes.get('prize', 'no prize')
    #     similarity = attributes.get('similarity', 'No Similarity')
    #     print(f"Node {node_id}: Prize = {prize}, Similarity = {similarity:.4f}")
        
        
    # for edge in G.edges(data=True):
    #     edge_id = edge[:2]
    #     attributes = edge[2]
    #     prize = attributes.get('prize', 'no prize')
    #     similarity = attributes.get('similarity', 'No Similarity')
    #     print(f"Edge {edge_id}: Prize = {prize}, Similarity = {similarity:.4f}, Edge Cost = {attributes.get('cost', 'No Cost')}, Edge Text = {attributes.get('relation', 'No Text')}")