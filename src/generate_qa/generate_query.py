import os
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..utils import call_llm
from ..config.generate_qa_config import LLM_CONFIG, PROCESSING_CONFIG, SEED_SAMPLE_SIZE


def process_query(index):
    """
    Process a single query generation request using LLM.
    
    Args:
        index (int): Index of the query being generated, used for error reporting
        
    Returns:
        dict: Generated query in JSON format containing question, type, placeholders, and DBpedia entities
        
    Note:
        This function will retry indefinitely until a valid query is generated
    """
    while True:
        system_prompt = "You are a helpful assistant designed to output JSON."
        user_prompt = open(os.path.join(os.path.dirname(__file__), "prompt.txt"), "r").read()
        try:
            query = call_llm(
                system_prompt=system_prompt, 
                user_prompt=user_prompt, 
                model_name=LLM_CONFIG["model_name"],
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"]
            )
            # remove the ```json and ``` from the query 
            query = query.strip().replace("```json", "").replace("```", "")
            query = json.loads(query)
            return query
        except Exception as e:
            print(f"Error generated query [index {index}]: {e}")
            continue
        

def multi_process_query(dataset_name:str, seed_sample_size:int = SEED_SAMPLE_SIZE):
    """
    Generate multiple queries in parallel using ThreadPoolExecutor.
    
    Args:
        dataset_name (str): Path where the generated dataset will be saved
        seed_sample_size (int, optional): Number of seed instructions to generate. 
            Note that the actual number of generated queries will be larger as 
            multiple queries are generated for each seed instruction. Defaults to 100.
    
    Returns:
        pd.DataFrame: DataFrame containing the generated queries with columns:
            - question: The generated question
            - type: Question type
            - placeholders: Dictionary of placeholders
            - dbpedia_entities: Dictionary of DBpedia entities
    """
    # if the dataset exists, read the dataset
    if os.path.exists(dataset_name):
        df = pd.read_csv(dataset_name, index_col=0)
        df["dbpedia_entities"] = df["dbpedia_entities"].apply(lambda x: eval(x))
        df["placeholders"] = df["placeholders"].apply(lambda x: eval(x))
    # if the dataset does not exist, create a new dataset
    else:
        results = []
        with ThreadPoolExecutor(max_workers=PROCESSING_CONFIG["max_workers"]) as executor:
            futures = [executor.submit(process_query, index) for index in range(seed_sample_size)]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating queries"):
                if future.result() is not None:
                    results.append(future.result())
            
    df = pd.DataFrame([item for sublist in results for item in sublist])
    df.to_csv(dataset_name, index=False)
    print(df.head())
    return df
