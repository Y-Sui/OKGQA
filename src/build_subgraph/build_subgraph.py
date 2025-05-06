import os
import json
import pandas as pd
import pickle
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm, trange
import urllib
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
import numpy as np
from collections import Counter
from src.utils import load_all_graphs, run_sparql
from preprocess import preprocess_graph

# set the maximum number of retries
MAX_RETRIES = 10
preprocess_graph_flag = False

# load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


class PPR_Utils:
    # PPR to prune the graph
    def __init__(self):
        # define the parameters
        self.alpha = 0.85
        self.tol = 1e-6
        self.max_iter = 100
        self.threshold = 1e-5  # threshold for pruning, if the node's PPR is less than the threshold, it will be pruned

    def calculate_ppr(self, G, central_node):
        # calcultae Personalized PageRank
        personalization = {node: 0 for node in G.nodes()}
        # set the central node weight to 1
        for node in list(central_node):
            personalization[node] = 1

        ppr = nx.pagerank(
            G,
            personalization=personalization,
            alpha=self.alpha,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        return ppr

    def prune_graph(self, G, central_node):
        pruned_G = nx.DiGraph()
        ppr = self.calculate_ppr(G, central_node)

        for node, score in ppr.items():
            if score >= self.threshold:
                pruned_G.add_node(node)

        for u, v, data in G.edges(data=True):
            if u in pruned_G and v in pruned_G:
                pruned_G.add_edge(u, v, **data)

        return pruned_G


def build_up_graph(rdfs):
    """_summary_

    Args:
        rdfs (dict): results from SPARQL query

    Returns:
        _type_: networkx graph
    """
    
    def extract_value(obj):
        if obj["type"] == "uri":
            return obj["value"].split("/")[-1].split("#")[-1]
        else:
            return obj["value"]
    
    # hold directed edges. self loops are allowed but multiple(parell) edges are not.
    G = nx.DiGraph()
    central_node = set()

    for result in rdfs["results"]["bindings"]:
        node = extract_value(result["entity"])
        central_node.add(node)

        first_hop_nei = extract_value(result["firstHopEntity"])
        second_hop_nei = extract_value(result["secondHopEntity"])
        r1 = extract_value(result["p"])
        r2 = extract_value(result["p2"])

        if node == first_hop_nei:
            # dbr:entity --> first_hop_nei --> second_hop_nei
            G.add_edge(node, first_hop_nei, relation=r1)
            G.add_edge(node, second_hop_nei, relation=r2)
        elif node == second_hop_nei:
            # dbr:entity -> first_hop_nei <- second_hop_nei
            G.add_edge(node, first_hop_nei, relation=r1)
            G.add_edge(second_hop_nei, first_hop_nei, relation=r2)
        elif first_hop_nei == second_hop_nei:
            # first_hop_nei -> dbr:entity -> second_hop_nei
            G.add_edge(first_hop_nei, node, relation=r1)
            G.add_edge(node, second_hop_nei, relation=r2)
        else:
            # first_hop_nei -> dbr:entity <- second_hop_nei
            G.add_edge(first_hop_nei, node, relation=r1)
            G.add_edge(second_hop_nei, first_hop_nei, relation=r2)

    return G, central_node

    
def retrieve_subgraph(index: int, entry_node: list, query: str, rerun=False):
    raw_graph_pth = f"subgraphs/raw/{index}.pkl"
    pruned_ppr_graph_pth = f"subgraphs/pruned_ppr/{index}.pkl"

    if rerun == False:
        if os.path.exists(raw_graph_pth) and os.path.exists(pruned_ppr_graph_pth):
            return index, True

    try:
        rdfs = run_sparql(entry_node)
        G, central_node = build_up_graph(rdfs)
        ppr_G = ppr.prune_graph(G, central_node)
        
        if preprocess_graph_flag:
            # build the ppr_G (generate embeddings)
            preprocess_graph(G=G, query_text=query, embedding_model="sbert", top_k_nodes=10, top_k_edges=10)
            pickle.dump(ppr_G, open(f"subgraphs/pruned_ppr_preprocessed/{index}.pkl", "wb"))
            return index, True
            
        pickle.dump(G, open(f"subgraphs/raw/{index}.pkl", "wb"))
        pickle.dump(ppr_G, open(f"subgraphs/pruned_ppr/{index}.pkl", "wb"))
        return index, True

    except Exception as e:
        print(f"Error: {e} occurred for subgraph {index}")
        return index, False


if __name__ == "__main__":
    # initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)

    df = pd.read_csv("query/filtered_questions_63a0f8a06513.csv", index_col=0)
    df["dbpedia_entities"] = df["dbpedia_entities"].apply(lambda x: eval(x))
    df["placeholders"] = df["placeholders"].apply(lambda x: eval(x))
    if isinstance(df["dbpedia_entities"].iloc[0], dict):
        df["dbpedia_entities_re"] = df["dbpedia_entities"].apply(
        lambda x: {k: v.split("/")[-1].split("#")[-1] for k, v in x.items()}
    )
    else:
        df["dbpedia_entities_re"] = df["dbpedia_entities"].apply(
            lambda x: {k: v for k, v in x.items()}
        )

    try:
        os.mkdir("subgraphs")
        os.mkdir("subgraphs/raw")
        os.mkdir("subgraphs/pruned_ppr")
        os.mkdir("subgraphs/pruned_ppr_preprocessed")
    except:
        pass

    error_subgraph_indices = []
    ppr = PPR_Utils()

    entry_nodes = []
    for entities in df["dbpedia_entities_re"]:
        entry_nodes.append(list(entities.values()))
    queries = df["question"]
    
    assert len(entry_nodes) == len(queries); print("Length of entry nodes and queries are not equal")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for index, (entry_node, query) in enumerate(zip(entry_nodes, queries)):
            futures.append(executor.submit(retrieve_subgraph, index, entry_node, query, True))
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Building subgraphs...",
            leave=True,
        ):
            index, flag = future.result()
            if flag == False:
                error_subgraph_indices.append(index)
                
    print(f"Lenght of error subgraphs: {len(error_subgraph_indices)}")
    with open("subgraphs/error_subgraph_indices.txt", "w") as f:
        for index in error_subgraph_indices:
            f.write(f"{index}\n")

    pruned_ppr_graphs = load_all_graphs("subgraphs/pruned_ppr/")
    idxs = []
    for g in pruned_ppr_graphs:
        idxs.append(g["idx"])
        
    df_valid = df.iloc[idxs]
    df_valid.to_csv("query/filtered_questions_63a0f8a06513_valid.csv", index=False)