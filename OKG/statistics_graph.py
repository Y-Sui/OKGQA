import tiktoken
import networkx as nx
import numpy as np
import pandas as pd
from utils import load_all_graphs


def calculate_statistics(G: nx.DiGraph):
    """_summary_

    Args:
       G (nx.DiGraph): _description_
    """

    # statistics
    print("Graph Statistics:")

    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    tokenizer = tiktoken.encoding_for_model("gpt-4o")

    edge_list = nx.generate_edgelist(G, data=True)
    tokens = len(tokenizer.encode("".join(edge_list)))

    number_of_nodes = G.number_of_nodes()
    number_of_edges = G.number_of_edges()

    # Degree statistics
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    total_degrees = [d for n, d in G.degree()]

    print(f"Average in-degree: {sum(in_degrees) / len(in_degrees):.2f}")
    print(f"Average out-degree: {sum(out_degrees) / len(out_degrees):.2f}")
    print(f"Average total degree: {sum(total_degrees) / len(total_degrees):.2f}")
    avg_in_degree = sum(in_degrees) / len(in_degrees)
    avg_out_degree = sum(out_degrees) / len(out_degrees)
    avg_total_degree = sum(total_degrees) / len(total_degrees)

    # Most common relations
    relations = [data['label'] for u, v, data in G.edges(data=True)]
    common_relations = Counter(relations).most_common(5)
    print("\nTop 5 most common relations:")
    for relation, count in common_relations:
        print(f"{relation}: {count}")

    # Number of connected components (for undirected version of the graph)
    undirected_G = G.to_undirected()
    num_components = nx.number_connected_components(undirected_G)
    print(f"\nNumber of connected components: {num_components}")

    # Largest connected component
    largest_cc = max(nx.connected_components(undirected_G), key=len)
    print(f"Size of the largest connected component: {len(largest_cc)}")

    # Check if the graph is a DAG (Directed Acyclic Graph)
    is_dag = nx.is_directed_acyclic_graph(G)
    print(f"\nIs the graph a DAG? {is_dag}")
    print(f"Is the graph strongly connected? {nx.is_strongly_connected(G)}")
    print(f"Is the graph weakly connected? {nx.is_weakly_connected(G)}")

    # Calculate the diameter of the largest connected component
    largest_cc_subgraph = undirected_G.subgraph(largest_cc)
    diameter = nx.diameter(largest_cc_subgraph)
    print(f"Diameter of the largest connected component: {diameter}")

    # Calculate the centrality of the top 5 nodes
    print("\nCentrality Measures (for top 5 nodes):")
    in_degree_centrality = nx.in_degree_centrality(G)
    print("Top 5 nodes by In-Degree Centrality:")
    for node, centrality in sorted(
        in_degree_centrality.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"{node}: {centrality:.4f}")

    out_degree_centrality = nx.out_degree_centrality(G)
    print("\nTop 5 nodes by Out-Degree Centrality:")
    for node, centrality in sorted(
        out_degree_centrality.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"{node}: {centrality:.4f}")

    # Betweenness Centrality (can be slow for large graphs)
    betweenness_centrality = nx.betweenness_centrality(G)
    print("\nTop 5 nodes by Betweenness Centrality:")
    for node, centrality in sorted(
        betweenness_centrality.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"{node}: {centrality:.4f}")

    # Clustering Coefficient
    clustering_coefficient = nx.average_clustering(G)
    print(f"\nAverage Clustering Coefficient: {clustering_coefficient:.4f}")

    # Shortest Paths
    print("\nShortest Path Statistics:")
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
    path_lengths = [
        length for paths in shortest_paths.values() for length in paths.values()
    ]
    print(f"Average Shortest Path Length: {np.mean(path_lengths):.2f}")
    print(f"Maximum Shortest Path Length (Diameter): {max(path_lengths)}")

    # Density
    density = nx.density(G)
    # print(f"\nGraph Density: {density:.4f}")

    return (
        tokens,
        number_of_nodes,
        number_of_edges,
        avg_in_degree,
        avg_out_degree,
        avg_total_degree,
        clustering_coefficient,
        density,
    )
    
    
def avg_statistics_of_G(df, Graphs):
    value = [0 for _ in range(8)]
    for G in Graphs:
        # calculate the average tokens, average nodes, average edges, average in-degree, out-degree, total-degree, average clustering coefficency, average graph density
        stats = calculate_statistics(G)
        value = [v + s for v, s in zip(value, stats)]

    avg_value = [v / len(df) for v in value]

    print(f"Average Tokens: {avg_value[0]:.2f}")
    print(f"Average Number of Nodes: {avg_value[1]:.2f}")
    print(f"Average Number of Edges: {avg_value[2]:.2f}")
    print(f"Average In-Degree: {avg_value[3]:.2f}")
    print(f"Average Out-Degree: {avg_value[4]:.2f}")
    print(f"Average Total Degree: {avg_value[5]:.2f}")
    print(f"Average Clustering Coefficient: {avg_value[6]:.2f}")
    print(f"Average Graph Density: {avg_value[7]:.2f}")

    return avg_value


if __name__ == "__main__":
    # Load dataframe
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
        
    # Remove invalid rows
    with open("subgraphs/error_subgraph_indices.txt", "r") as f:
        error_subgraph_indices = [int(index) for index in f.readlines()]
    df = df.drop(error_subgraph_indices)

    # Load all graphs
    raw_graphs = load_all_graphs("subgraphs/raw/")
    pruned_ppr_graphs = load_all_graphs("subgraphs/pruned_ppr/")

    avg_statistics_of_G(df, raw_graphs)
    avg_statistics_of_G(df, pruned_ppr_graphs)