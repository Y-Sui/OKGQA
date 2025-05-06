import os
import networkx as nx
import numpy as np
import pickle
import pandas as pd
from openai import OpenAI
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv
from src.utils import load_all_graphs
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

def compute_embeddings(graph, query, model="sbert"):
    """
    Retrieve the embedding for a given text using OpenAI's embedding API or language model.

    Parameters:
    - text (str): The input text to embed.
    - model (str): The embedding model to use.

    Returns:
    - List[float]: The embedding vector.
    """
    if model == "text-embedding-ada-002":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        q_embedding = client.embeddings.create(input=query, model=model).data.embedding
        
        for node in graph.nodes:
            embedding = client.embeddings.create(input=node, model=model).data.embedding
            graph.nodes[node]['embedding'] = embedding
            
        for u, v, data in graph.edges(data=True):
            edge_text = data.get('relation', f"{u}->{v}")
            embedding = client.embeddings.create(input=edge_text, model=model).data.embedding
            graph.edges[u, v]['embedding'] = embedding
            
    elif model == "sbert":
        model = SentenceTransformer('all-MiniLM-L6-v2')
        q_embedding = model.encode(query)
    
        for node in graph.nodes:
            embedding = model.encode(node)
            graph.nodes[node]['embedding'] = embedding
            
        for u, v, data in graph.edges(data=True):
            edge_text = data.get('relation', f"{u}->{v}")
            embedding = model.encode(edge_text)
            graph.edges[u, v]['embedding'] = embedding
        
    return q_embedding

def compute_similarity_scores(graph, q_embedding):
    """
    Compute cosine similarity scores between the query and nodes/edges.
    """
    q_norm = np.linalg.norm(q_embedding)
    
    # Compute similarity scores for nodes
    for node, data in graph.nodes(data=True):
        emb = data['embedding']
        score = np.dot(q_embedding, emb) / (q_norm * np.linalg.norm(emb))
        graph.nodes[node]['similarity'] = score
        
    # Compute similarity scores for edges
    for u, v, data in graph.edges(data=True):
        emb = data['embedding']
        score = np.dot(q_embedding, emb) / (q_norm * np.linalg.norm(emb))
        graph.edges[u, v]['similarity'] = score

def assign_prizes(graph, top_k_nodes: int, top_k_edges: int):
    """
    Assign prizes to the top-k nodes and edges based on similarity scores.
    """
    # Assign prizes to top-k nodes
    node_similarities = [(node, data['similarity']) for node, data in graph.nodes(data=True)]
    node_similarities.sort(key=lambda x: x[1], reverse=True)
    for rank, (node, _) in enumerate(node_similarities):
        prize = max(top_k_nodes - rank, 0)
        graph.nodes[node]['prize'] = prize
    # Assign prizes to top-k edges
    edge_similarities = [((u, v), data['similarity']) for u, v, data in graph.edges(data=True)]
    edge_similarities.sort(key=lambda x: x[1], reverse=True)
    for rank, ((u, v), _) in enumerate(edge_similarities):
        prize = max(top_k_edges - rank, 0)
        graph.edges[u, v]['prize'] = prize

def assign_costs(graph, default_cost=1.0):
    """
    Assign default costs to edges.
    """
    for u, v, data in graph.edges(data=True):
        data['cost'] = default_cost

def preprocess_graph(
    G: nx.Graph,
    query_text: str,
    top_k_nodes: int,
    top_k_edges: int,
    default_edge_cost: float = 1.0,
    embedding_model: str = "sbert"
):
    """
    Preprocess the graph by computing embeddings, similarity scores, and assigning prizes.
    
    Parameters:
    - G (nx.Graph): The input graph.
    - query_text (str): The query text.
    - top_k_nodes (int): The number of top nodes to assign prizes to.
    - top_k_edges (int): The number of top edges to assign prizes to.
    - output_pkl_path (str): The output path to save the processed graph.
    - default_edge_cost (float): The default edge cost.
    - embedding_model (str): The embedding model to use.
    """
    # Compute embeddings
    q_embedding = compute_embeddings(G, query_text, model=embedding_model)
    
    # Compute similarity scores
    compute_similarity_scores(G, q_embedding)
    
    # Assign prizes
    assign_prizes(G, top_k_nodes, top_k_edges)
    
    # Assign costs
    assign_costs(G, default_cost=default_edge_cost)
    
    
def main():
    try:
        os.mkdir("subgraphs/pruned_ppr_preprocessed")
    except:
        pass
    
    df = pd.read_csv("query/filtered_questions_63a0f8a06513_valid.csv")
    pruned_ppr_graphs = load_all_graphs("subgraphs/pruned_ppr/")
    graphs = [item['graph'] for item in pruned_ppr_graphs]
    idxs = [item['idx'] for item in pruned_ppr_graphs]
    questions = df["question"]
    
    for i, (query, G) in tqdm(enumerate(zip(questions, graphs)), desc="preprocessing graphs", total=len(questions)):
        preprocess_graph(G=G, query_text=query, embedding_model="sbert", top_k_nodes=10, top_k_edges=10)
        pickle.dump(G, open(f"subgraphs/pruned_ppr_preprocessed/{idxs[i]}.pkl", "wb"))
        
if __name__ == "__main__":
    main()