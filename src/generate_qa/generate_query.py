import os
import json
import pandas as pd
import requests
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import ast
from collections import Counter
from SPARQLWrapper import SPARQLWrapper, JSON
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..utils import call_llm
from datetime import datetime


system_prompt = """
    You are a helpful assistant designed to output JSON.
"""

user_prompt = open(os.path.join(os.path.dirname(__file__), "prompt.txt"), "r").read()

def process_query(index):
    while True:
        try:
            query = call_llm(system_prompt, user_prompt)
            # remove the ```json and ``` from the query 
            query = query.strip().replace("```json", "").replace("```", "")
            query = json.loads(query)
            return query
        except Exception as e:
            print(f"Error generated query [index {index}]: {e}")
            print(f"Error generated query: {query}")
            continue
        

def multi_process_query(dataset_name:str, sample_size:int = 100):
    """
    sample_size: the number of the seed instruction to generate (noted that the number of the generated queries will be larger than the sample_size, as we generate five queries for each seed instruction)
    
    dataset_name: the path to save the generated queries
    """
    # if the dataset exists, read the dataset
    if os.path.exists(dataset_name):
        df = pd.read_csv(dataset_name, index_col=0)
        df["dbpedia_entities"] = df["dbpedia_entities"].apply(lambda x: eval(x))
        df["placeholders"] = df["placeholders"].apply(lambda x: eval(x))
    # if the dataset does not exist, create a new dataset
    else:
        results = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_query, index) for index in range(sample_size)]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating queries"):
                if future.result() is not None:
                    results.append(future.result())
            
    df = pd.DataFrame([item for sublist in results for item in sublist])
    df.to_csv(dataset_name, index=False)
    print(df.head())
    return df


def main():
    sample_size = 100
    dir_path = "/mnt/250T_ceph/tristanysui/okgqa"
    dataset_name = dir_path + f"/questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sample_size}.csv"
    multi_process_query(dataset_name, sample_size=sample_size)

if __name__ == "__main__":
    main()

