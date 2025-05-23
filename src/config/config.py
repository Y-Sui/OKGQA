import os
from datetime import datetime
# Base directories
BASE_DIR = "/mnt/250T_ceph/tristanysui/okgqa"
WIKI_DIR = os.path.join(BASE_DIR, "wikipedia")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
QUERY_DIR = os.path.join(BASE_DIR, "queries")
SUBGRAPH_DIR = os.path.join(BASE_DIR, "subgraphs")

# Generation parameters
SEED_SAMPLE_SIZE = 5
RE_SAMPLE = False
TIMESTAMP_FORMAT = '%Y%m%d'
TIMESTAMP = datetime.now().strftime(TIMESTAMP_FORMAT)

# LLM parameters
LLM_CONFIG = {
    "model_name": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 4000  
}

# HTTP request parameters
HTTP_CONFIG = {
    "headers": {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive'
    },
    "timeout": 5,
    "verify_ssl": True
}

# Wikipedia API parameters
WIKI_CONFIG = {
    "user_agent": "OpenKGQA/0.0 (yuansui08@gmail.com)",
    "language": "en",
    "sent_split": False,
    "rerun": True
}

# Processing parameters
PROCESSING_CONFIG = {
    "max_workers": os.cpu_count(),
    "wiki_workers": max(1, os.cpu_count() - 1)
}

# Subgraph parameters
SUBGRAPH_CONFIG = {
    "raw_dir": os.path.join(SUBGRAPH_DIR, "raw"),
    "pruned_ppr_dir": os.path.join(SUBGRAPH_DIR, "pruned_ppr"),
    "pruned_ppr_init_dir": os.path.join(SUBGRAPH_DIR, "pruned_ppr_init"),
    "error_indices_file": os.path.join(SUBGRAPH_DIR, "error_subgraph_indices.txt"),
    "perturbed_graphs_dir": os.path.join(SUBGRAPH_DIR, "perturbed_graphs"),
    "preprocess_graph_flag": False,
    "ppr_params": {
        "alpha": 0.85,
        "tol": 1e-6,
        "max_iter": 100,
        "threshold": 1e-5
    },
    "preprocess_params": {
        "top_k_nodes": 10,
        "top_k_edges": 10,
        "default_edge_cost": 1.0,
        "embedding_model": "text-embedding-3-small"
    },
    "link_prediction_model_path": os.path.join(SUBGRAPH_DIR, "link_prediction_model"),
    "re_train_link_prediction_model": True
}