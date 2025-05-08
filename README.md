# OKGQA: Open Knowledge Graph Question Answering

This repository contains the implementation of paper ["Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study over Open-ended Question Answering"](https://arxiv.org/abs/2410.08085). This paper introduces OKGQA, a new benchmark for evaluating Knowledge Graph-enhanced LLMs in open-ended question answering, focusing on reducing hallucinations and improving reasoning.

## Overview

OKGQA is a comprehensive benchmark that combines Knowledge Graphs (KGs) with Large Language Models (LLMs) to enhance the trustworthiness of open-ended question answering. The system leverages DBpedia as its primary knowledge source and implements various techniques to improve answer quality and reduce hallucinations.

## Key Features

- **Open-ended QA**: Focuses on real-world, free-form questions answering requiring complex reasoning
- **Diverse Question Types**: Includes 10+ categories of QA to mirror practical scenarios
- **Noise-resilient Testing (OKGQA-P)**: Evaluates robustness against perturbed/contaminated KGs
- **Hallucination Metrics**: Measures factual accuracy via FActScore and SAFE, alongside LLM-as-judge for answer quality (relevance, correctness, etc.)
- **Knowledge Graph Integration**: Seamless integration with DBpedia for factual grounding

## Project Structure

```
src/
├── build_subgraph/          # Subgraph construction and manipulation
│   ├── build_subgraph.py    # Core subgraph building functionality
│   ├── g_retriever.py       # Graph retrieval utilities
│   ├── perturb_kgs.py       # Knowledge graph perturbation
│   ├── preprocess.py        # Data preprocessing
│   ├── score_function_sG.py # Scoring functions for subgraphs
│   └── statistics_graph.py  # Graph statistics calculation
├── generate_qa/             # Question-Answer generation
│   ├── main.py             # Main QA generation script
│   ├── generate_query.py   # SPARQL query generation
│   ├── retrieve_wikipedia.py# Wikipedia data retrieval
│   ├── post_process.py     # Post-processing utilities
│   ├── calculate_stat.py   # Statistics calculation
│   └── prompt.txt          # LLM prompts for QA generation
├── config/                 # Configuration files
│   └── config.py          # Main configuration settings
├── utils.py               # Utility functions
└── eval/                  # Evaluation scripts
```

### Component Details

#### build_subgraph/
- `build_subgraph.py`: Constructs knowledge subgraphs from DBpedia data
- `g_retriever.py`: Implements graph retrieval algorithms
- `perturb_kgs.py`: Handles knowledge graph perturbation for robustness testing
- `preprocess.py`: Preprocesses raw data for graph construction
- `score_function_sG.py`: Implements scoring functions for subgraph quality
- `statistics_graph.py`: Calculates graph statistics and metrics

#### generate_qa/
- `main.py`: Orchestrates the QA pair generation process
- `generate_query.py`: Generates SPARQL queries for knowledge retrieval
- `retrieve_wikipedia.py`: Fetches relevant Wikipedia content
- `post_process.py`: Post-processes generated QA pairs
- `calculate_stat.py`: Computes statistics for generated QA pairs
- `prompt.txt`: Contains LLM prompts for QA generation

## Requirements

- Python 3.8+
- OpenAI API key
- NetworkX (>=3.0)
- SPARQLWrapper (>=2.0.0)
- python-dotenv (>=1.0.0)
- Transformers
- NumPy (>=1.24.0)
- Pandas (>=2.0.0)
- tqdm (>=4.65.0)
- requests (>=2.31.0)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Y-Sui/OKGQA.git
cd okgqa
```

2. Install dependencies:
```bash
conda env create -f okgqa_39_env_dev.yml
```

3. Set up environment variables:
Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Configuration

Set up the configuration for generation and perturbation:
```bash
vim src/config/config.py
```

Key configuration parameters include:
- API keys and endpoints
- Model parameters
- Graph construction settings
- QA generation parameters
- Evaluation metrics

### Running the Pipeline

1. Generate QA pairs:
```bash
python -m src.generate_qa.main
```

2. Generate subgraphs:
```bash
python -m src.build_subgraph.build_subgraph
```


3. Generate perturbed subgraphs (for robustness testing in OKGQA-P):
```bash
python -m src.build_subgraph.perturb_kgs
```

## Evaluation

The system includes comprehensive evaluation metrics:
- FActScore for factual accuracy
- SAFE for answer quality
- LLM-as-judge for relevance and correctness
- Custom metrics for KG integration effectiveness

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:

```
@misc{sui2025knowledgegraphsmakelarge,
      title={Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering}, 
      author={Yuan Sui and Yufei He and Zifeng Ding and Bryan Hooi},
      year={2025},
      eprint={2410.08085},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.08085}, 
}
```