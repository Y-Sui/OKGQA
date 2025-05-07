import os

# Base directories
BASE_DIR = "/mnt/250T_ceph/tristanysui/okgqa"
WIKI_DIR = os.path.join(BASE_DIR, "wikipedia")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Generation parameters
SEED_SAMPLE_SIZE = 5
RE_SAMPLE = False
TIMESTAMP_FORMAT = '%Y%m%d'

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

# File paths
PATHS = {
    "queries_dir": os.path.join(BASE_DIR, "queries"),
    "plots_dir": os.path.join(BASE_DIR, "plots"),
} 