import os
import pandas as pd
import pickle
import networkx as nx
from ..utils import load_all_graphs, run_sparql
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .preprocess import preprocess_graph
from ..config.config import SUBGRAPH_CONFIG, PROCESSING_CONFIG, QUERY_DIR, TIMESTAMP, SEED_SAMPLE_SIZE
from .statistics_graph import avg_statistics_of_G


class PPR_Utils:
    # PPR to prune the graph
    def __init__(self):
        # define the parameters from config
        self.alpha = SUBGRAPH_CONFIG["ppr_params"]["alpha"]
        self.tol = SUBGRAPH_CONFIG["ppr_params"]["tol"]
        self.max_iter = SUBGRAPH_CONFIG["ppr_params"]["max_iter"]
        self.threshold = SUBGRAPH_CONFIG["ppr_params"]["threshold"]

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
    raw_graph_pth = os.path.join(SUBGRAPH_CONFIG["raw_dir"], f"{index}.pkl")
    pruned_ppr_graph_pth = os.path.join(SUBGRAPH_CONFIG["pruned_ppr_dir"], f"{index}.pkl")
    
    if not rerun:
        if os.path.exists(raw_graph_pth) and os.path.exists(pruned_ppr_graph_pth):
            return index, True

    try:
        # run the SPARQL query
        rdfs = run_sparql(entry_node)
    except Exception as e:
        print(f"SPARQL query failed for subgraph {index}: {str(e)}")
        return index, False
    
    # build the graph
    G, central_node = build_up_graph(rdfs)
    print(f"Built graph for subgraph {index}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # prune the graph
    ppr = PPR_Utils()
    ppr_G = ppr.prune_graph(G, central_node)
    print(f"Pruned graph for subgraph {index}: {ppr_G.number_of_nodes()} nodes, {ppr_G.number_of_edges()} edges")
    
    # save the raw and pruned graphs
    pickle.dump(G, open(raw_graph_pth, "wb"))
    pickle.dump(ppr_G, open(pruned_ppr_graph_pth, "wb"))

    if SUBGRAPH_CONFIG["preprocess_graph_flag"]:
        pruned_ppr_preprocessed_graph_pth = os.path.join(
            SUBGRAPH_CONFIG["pruned_ppr_init_dir"], 
            f"{index}.pkl"
        )
        # build the ppr_G (generate embeddings) and save the preprocessed graph
        preprocess_graph(
            G=ppr_G, 
            query_text=query, 
            embedding_model=SUBGRAPH_CONFIG["preprocess_params"]["embedding_model"],
            top_k_nodes=SUBGRAPH_CONFIG["preprocess_params"]["top_k_nodes"],
            top_k_edges=SUBGRAPH_CONFIG["preprocess_params"]["top_k_edges"]
        )
        pickle.dump(ppr_G, open(pruned_ppr_preprocessed_graph_pth, "wb"))
        return index, True
        
    return index, True


def main():
    df = pd.read_csv(os.path.join(QUERY_DIR, f"questions_{TIMESTAMP}_{SEED_SAMPLE_SIZE}_post_processed.csv"), index_col=0)
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

    # Create directories
    for dir_path in [
        SUBGRAPH_CONFIG["raw_dir"],
        SUBGRAPH_CONFIG["pruned_ppr_dir"],
        SUBGRAPH_CONFIG["pruned_ppr_init_dir"],
    ]:
        os.makedirs(dir_path, exist_ok=True)

    error_subgraph_indices = []
    entry_nodes = []
    for entities in df["dbpedia_entities_re"]:
        entry_nodes.append(list(entities.values()))
    queries = df["question"]
    
    assert len(entry_nodes) == len(queries), "Length of entry nodes and queries are not equal"

    with ThreadPoolExecutor(max_workers=PROCESSING_CONFIG["max_workers"]) as executor:
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
            if not flag:
                error_subgraph_indices.append(index)
                
    print(f"Length of error subgraphs: {len(error_subgraph_indices)}")
    with open(SUBGRAPH_CONFIG["error_indices_file"], "w") as f:
        for index in error_subgraph_indices:
            f.write(f"{index}\n")

    pruned_ppr_graphs = load_all_graphs(SUBGRAPH_CONFIG["pruned_ppr_dir"])
    idxs = []
    for g in pruned_ppr_graphs:
        idxs.append(g["idx"])
    raw_graphs = load_all_graphs(SUBGRAPH_CONFIG["raw_dir"])
    # avg_statistics_of_G(df, raw_graphs)
    avg_statistics_of_G(df, pruned_ppr_graphs)
        
    df_valid = df.iloc[idxs]
    df_valid.to_csv(os.path.join(QUERY_DIR, f"questions_{TIMESTAMP}_{SEED_SAMPLE_SIZE}_final.csv"), index=False)
    
    
    
if __name__ == "__main__":
    main()