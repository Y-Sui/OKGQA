import pickle
import os
import networkx as nx
from SPARQLWrapper import SPARQLWrapper, JSON


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

q_prefix = """
  Generate 5 open-ended questions about different types: character description, event description, cause explanation, relationship explanation, trend prediction, outcome prediction, contrast analysis, historical comparison, evaluation and reflection, and application and practice. Some templates are provided below:
  character description: describe a [person]'s significant contributions during their career. Example: Please describe Albert Einstein’s contributions to the field of physics.
  event description: Provide a detailed description of the background and course of an [event]. Example: Please provide a detailed description of the background and course of the French Revolution.
  cause explanation: Why did [person] take [action] at [time]? Example: Why did Nixon choose to resign from the presidency in 1974?
  relationship explanation: Explain the relationship between [entity A] and [entity B] and its significance. Example: Explain the relationship between Alexander the Great and Aristotle and its significance.
  trend prediction: Based on the historical behavior of [entity], what do you think it might do in the future? Example: Based on Tesla’s historical behavior, in which fields do you think it might innovate in the future?
  outcome prediction: Based on the current situation, how do you predict [event] will develop? Example: Based on the current international situation, how do you predict climate change policies will develop?
  contrast analysis: Compare and contrast the similarities and differences between [entity A] and [entity B] in [aspect]. Example: Compare and contrast the leadership styles of Steve Jobs and Bill Gates.
  historical comparison: Compare the impact of [historical event A] and [historical event B]. Example: Compare the impact of World War I and World War II on the global order
  evaluation and reflection: How do you evaluate the impact of [person/event] on [field]? Please explain your viewpoint. Example: How do you evaluate Martin Luther King’s impact on the civil rights movement? Please explain your viewpoint.
  application and practice: How do you think [theory/method] can be applied to [practical issue]? Please provide specific suggestions. Example: How do you think machine learning technology can be applied to medical diagnostics? Please provide specific suggestions.
  Generate the questions, the type of the questions, the placeholders, the naturalness of your generated questions (choose from high, medium, and unnatural), the difficulty of the generated questions (choose from hard, medium and easy) and dbpedia_entities (link the placeholders to dbpedia entities) in JSON format.
"""

q_example = """
    The following is an example of the output format:
    {'question': 'Compare and contrast the similarities and differences between the Apple iPhone and Samsung Galaxy in terms of user interface design.',
    'type': 'contrast analysis',
    'placeholders': {'entity A': 'Apple iPhone',
    'entity B': 'Samsung Galaxy',
    'aspect': 'user interface design'},
    'naturalness': 'high',
    'difficulty': 'medium',
    'dbpedia_entities': {'entity A': 'http://dbpedia.org/resource/IPhone',
    'entity B': 'http://dbpedia.org/resource/Samsung_Galaxy'}
    }
"""

def run_sparql(entities):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

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


def load_all_graphs(raw_graph_dir, sample_size=None):
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