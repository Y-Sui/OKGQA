import pickle
import os
import time
import networkx as nx
from SPARQLWrapper import SPARQLWrapper, JSON
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# SPARQL query for single entity
single_entity_query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbr: <http://dbpedia.org/resource/>
SELECT DISTINCT ?entity ?p ?firstHopEntity ?p2 ?secondHopEntity
WHERE {{
  {{
    dbr:{entity} ?p ?firstHopEntity.
    ?firstHopEntity ?p2 ?secondHopEntity.
    BIND(dbr:{entity} AS ?entity)
  }} UNION {{
    dbr:{entity} ?p ?firstHopEntity.
    ?secondHopEntity ?p2 ?firstHopEntity.
    BIND(dbr:{entity} AS ?entity)
  }} UNION {{
    ?firstHopEntity ?p dbr:{entity}.
    ?firstHopEntity ?p2 ?secondHopEntity.
    BIND(dbr:{entity} AS ?entity)
  }} UNION {{
    ?firstHopEntity ?p dbr:{entity}.
    ?secondHopEntity ?p2 ?firstHopEntity.
    BIND(dbr:{entity} AS ?entity)
  }}
}}
"""

# SPARQL query for two entities
two_entity_query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbr: <http://dbpedia.org/resource/>
SELECT DISTINCT ?entity ?p ?firstHopEntity ?p2 ?secondHopEntity
WHERE {{
    {{
    dbr:{entity1} ?p ?firstHopEntity.
    ?firstHopEntity ?p2 ?secondHopEntity.
    BIND(dbr:{entity1} AS ?entity)
    }} UNION {{
    dbr:{entity1} ?p ?firstHopEntity.
    ?secondHopEntity ?p2 ?firstHopEntity.
    BIND(dbr:{entity1} AS ?entity)
    }} UNION {{
    ?firstHopEntity ?p dbr:{entity1}.
    ?firstHopEntity ?p2 ?secondHopEntity.
    BIND(dbr:{entity1} AS ?entity)
    }} UNION {{
    ?firstHopEntity ?p dbr:{entity1}.
    ?secondHopEntity ?p2 ?firstHopEntity.
    BIND(dbr:{entity1} AS ?entity)
    }} UNION {{
    dbr:{entity2} ?p ?firstHopEntity.
    ?firstHopEntity ?p2 ?secondHopEntity.
    BIND(dbr:{entity2} AS ?entity)
    }} UNION {{
    dbr:{entity2} ?p ?firstHopEntity.
    ?secondHopEntity ?p2 ?firstHopEntity.
    BIND(dbr:{entity2} AS ?entity)
    }} UNION {{
    ?firstHopEntity ?p dbr:{entity2}.
    ?firstHopEntity ?p2 ?secondHopEntity.
    BIND(dbr:{entity2} AS ?entity)
    }} UNION {{
    ?firstHopEntity ?p dbr:{entity2}.
    ?secondHopEntity ?p2 ?firstHopEntity.
    BIND(dbr:{entity2} AS ?entity)
    }}
}}
"""

# SPARQL query for three entities

three_entity_query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbr: <http://dbpedia.org/resource/>
SELECT DISTINCT ?entity ?p ?firstHopEntity ?p2 ?secondHopEntity
WHERE {{
  {{
    dbr:{entity1} ?p ?firstHopEntity.
    ?firstHopEntity ?p2 ?secondHopEntity.
    BIND(dbr:{entity1} AS ?entity)
  }} UNION {{
    dbr:{entity1} ?p ?firstHopEntity.
    ?secondHopEntity ?p2 ?firstHopEntity.
    BIND(dbr:{entity1} AS ?entity)
  }} UNION {{
    ?firstHopEntity ?p dbr:{entity1}.
    ?firstHopEntity ?p2 ?secondHopEntity.
    BIND(dbr:{entity1} AS ?entity)
  }} UNION {{
    ?firstHopEntity ?p dbr:{entity1}.
    ?secondHopEntity ?p2 ?firstHopEntity.
    BIND(dbr:{entity1} AS ?entity)
  }} UNION {{
    dbr:{entity2} ?p ?firstHopEntity.
    ?firstHopEntity ?p2 ?secondHopEntity.
    BIND(dbr:{entity2} AS ?entity)
  }} UNION {{
    dbr:{entity2} ?p ?firstHopEntity.
    ?secondHopEntity ?p2 ?firstHopEntity.
    BIND(dbr:{entity2} AS ?entity)
  }} UNION {{
    ?firstHopEntity ?p dbr:{entity2}.
    ?firstHopEntity ?p2 ?secondHopEntity.
    BIND(dbr:{entity2} AS ?entity)
  }} UNION {{
    ?firstHopEntity ?p dbr:{entity2}.
    ?secondHopEntity ?p2 ?firstHopEntity.
    BIND(dbr:{entity2} AS ?entity)
  }} UNION {{
    dbr:{entity3} ?p ?firstHopEntity.
    ?firstHopEntity ?p2 ?secondHopEntity.
    BIND(dbr:{entity3} AS ?entity)
  }} UNION {{
    dbr:{entity3} ?p ?firstHopEntity.
    ?secondHopEntity ?p2 ?firstHopEntity.
    BIND(dbr:{entity3} AS ?entity)
  }} UNION {{
    ?firstHopEntity ?p dbr:{entity3}.
    ?firstHopEntity ?p2 ?secondHopEntity.
    BIND(dbr:{entity3} AS ?entity)
  }} UNION {{
    ?firstHopEntity ?p dbr:{entity3}.
    ?secondHopEntity ?p2 ?firstHopEntity.
    BIND(dbr:{entity3} AS ?entity)
  }}
}}
"""

def run_sparql(entities):
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")

    if len(entities) == 1:
        query = single_entity_query
        query = query.format(entity=entities[0])
    elif len(entities) == 2:
        query = two_entity_query
        query = query.format(entity1=entities[0], entity2=entities[1])
    elif len(entities) == 3:
        query = three_entity_query
        query = query.format(
            entity1=entities[0], entity2=entities[1], entity3=entities[2]
        )

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results


def load_all_graphs(raw_graph_dir:str, sample_size:int=None):
    """
    Loads all pickle graph files from the specified directory.

    Parameters:
    - raw_graph_dir (str): Path to the directory containing raw graph pickle files.

    Returns:
    - graphs (dict): A dictionary where keys are filenames and values are the loaded graph objects.
    """
    graphs = []

    # Check if the directory exists
    if not os.path.isdir(raw_graph_dir):
        raise FileNotFoundError(f"The directory '{raw_graph_dir}' does not exist.")

    # Iterate through all files in the directory
    for filename in os.listdir(raw_graph_dir):
        filepath = os.path.join(raw_graph_dir, filename)

        # Check if it's a file and has a .pkl extension
        if os.path.isfile(filepath) and filename.endswith('.pkl'):
            try:
                with open(filepath, 'rb') as file:
                    graph = pickle.load(file)
                    
                    # Optional: If using NetworkX, verify it's a graph
                    if isinstance(graph, nx.Graph):
                        # print(f"Loaded NetworkX graph '{filename}' with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
                        print(f"Loaded NetworkX graph '{filename}'.")
                    else:
                        print(f"Loaded object from '{filename}'. It may not be a NetworkX graph.")
                    
                    if graph:
                      graphs.append(
                        {
                          "idx": filename.split(".")[0],
                          "graph": graph
                        }
                      )
                    if sample_size is not None and len(graphs) >= sample_size:
                        break
            except Exception as e:
                print(f"Failed to load '{filename}': {e}")

    return graphs
  
  
def call_llm(system_prompt: str, user_prompt: str, model_name: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 4000):
    attempt = 0
    while attempt < 10:
        try:
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            print(f"Response: {response}")
            attempt += 1
            time.sleep(1) 
    return None
