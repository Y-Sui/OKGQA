import os
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..utils import call_llm
from datetime import datetime
from ..config.generate_qa_config import LLM_CONFIG, PROCESSING_CONFIG, PATHS, SEED_SAMPLE_SIZE


def process_query(index):
    while True:
        try:
            query = call_llm(LLM_CONFIG["system_prompt"], open(PATHS["prompt_file"], "r").read(), model=LLM_CONFIG["model"])
            # remove the ```json and ``` from the query 
            query = query.strip().replace("```json", "").replace("```", "")
            query = json.loads(query)
            return query
        except Exception as e:
            print(f"Error generated query [index {index}]: {e}")
            print(f"Error generated query: {query}")
            continue
        

def multi_process_query(dataset_name:str, seed_sample_size:int = 100):
    """
    seed_sample_size: the number of the seed instruction to generate (noted that the number of the generated queries
    will be larger than the sample_size, as we generate five queries for each seed instruction)
    
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
        with ThreadPoolExecutor(max_workers=PROCESSING_CONFIG["max_workers"]) as executor:
            futures = [executor.submit(process_query, index) for index in range(seed_sample_size)]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating queries"):
                if future.result() is not None:
                    results.append(future.result())
            
    df = pd.DataFrame([item for sublist in results for item in sublist])
    df.to_csv(dataset_name, index=False)
    print(df.head())
    return df


def main():
    dataset_name = os.path.join(PATHS["queries_dir"], f"questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{SEED_SAMPLE_SIZE}.csv")
    multi_process_query(dataset_name, sample_size=SEED_SAMPLE_SIZE)

if __name__ == "__main__":
    main()

