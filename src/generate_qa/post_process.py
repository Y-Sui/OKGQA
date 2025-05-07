import pandas as pd
import os
import requests
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .retrieve_wikipedia import get_wikipedia_pages
from ..config.generate_qa_config import HTTP_CONFIG, PROCESSING_CONFIG

    
def check_url(url: str) -> bool:
    """
    Check if a DBpedia URL is valid and accessible.
    
    Args:
        url (str): The DBpedia URL to check
        
    Returns:
        bool: True if the URL is valid and accessible, False otherwise
        
    Note:
        Uses both HEAD and GET requests with proper headers and SSL verification.
        Falls back to GET request if HEAD is not allowed.
    """
    try:
        # First try HEAD request
        response = requests.head(
            url.replace("http://", "https://"),
            headers=HTTP_CONFIG["headers"],
            timeout=HTTP_CONFIG["timeout"],
            allow_redirects=True,
            verify=HTTP_CONFIG["verify_ssl"]
        )
        
        # Fallback to GET if HEAD is not allowed
        if response.status_code == 405:
            response = requests.get(
                url.replace("http://", "https://"),
                headers=HTTP_CONFIG["headers"],
                timeout=HTTP_CONFIG["timeout"],
                allow_redirects=True,
                verify=HTTP_CONFIG["verify_ssl"],
                stream=True  # Don't download content
            )
            
        return response.ok

    except requests.exceptions.SSLError:
        return False
    except requests.exceptions.RequestException:
        return False


def post_process(df: pd.DataFrame):
    """
    Post-process the generated queries to remove duplicates and validate format.
    
    Args:
        df (pd.DataFrame): DataFrame containing raw generated queries
        
    Returns:
        pd.DataFrame: Filtered DataFrame with unique queries based on type, 
            placeholders, and DBpedia entities
        
    Note:
        Uses frozenset to make dictionaries hashable for duplicate detection
    """
    unique_sample = set()
    rows_to_keep = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Post-processing samples"):
        type = row["type"]
        placeholders = row["placeholders"]
        dbpedia_entities = row["dbpedia_entities"]
        try:
            # frozenset is hashable
            placeholders = frozenset(row["placeholders"].items())
            dbpedia_entities = frozenset(row["dbpedia_entities"].items())
            if (type, placeholders, dbpedia_entities) not in unique_sample:
                unique_sample.add((type, placeholders, dbpedia_entities))
                rows_to_keep.append(i)

        except Exception as e:
            # print(f"Error processing row {i}: {e}")
            # print("placeholders: ", placeholders)
            # print((row["type"], placeholders, dbpedia_entities))
            # print("question: ", question)
            pass

    print(f"Number of unique samples: {len(unique_sample)}")
    df_filtered = df.iloc[rows_to_keep].reset_index(drop=True)

    return df_filtered 


def verify_and_filter_entities(df: pd.DataFrame):
    """
    Verify and filter entities by checking if their DBpedia URLs are valid.
    
    Args:
        df (pd.DataFrame): DataFrame containing queries with DBpedia entities
        
    Returns:
        pd.DataFrame: Filtered DataFrame containing only queries with valid entities
        
    Note:
        Uses parallel processing to check URLs efficiently
    """
    valid_rows = set()
    print(f"Total rows to process: {len(df)}")

    # use ThreadPoolExecutor for parallel calling of check_url()
    with ThreadPoolExecutor(max_workers=PROCESSING_CONFIG["max_workers"]) as executor:
        futures = {}
        for index, row in df.iterrows():
            for key, url in row['dbpedia_entities'].items():
                future = executor.submit(check_url, url)
                futures[future] = (index, key)

        url_results = {}  # store the results of the futures
        # process the futures as they are completed
        for future in tqdm(as_completed(futures), total=len(futures), desc="Verifying URLs"):
            result = future.result()
            url_results[futures[future]] = result

    # iterate through the DataFrame rows to filter out invalid ones
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Filtering invalid rows"):
        valid = True
        for key, url in row['dbpedia_entities'].items():    
            # if any URL is invalid, set valid to False and break
            if not url_results[(index, key)]:
                valid = False
                # print(f"Row {index} invalid due to URL: {url}")
                break
        if valid:
            valid_rows.add(index)

    print(f"Number of valid rows: {len(valid_rows)}")
    filtered_df = df.loc[list(valid_rows)].reset_index(drop=True)
    return filtered_df


def retrieve_wikipedia_pages(df: pd.DataFrame, wiki_dir: str):
    """
    Retrieve Wikipedia pages for all entities in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing queries with DBpedia entities
        
    Note:
        Extracts entity names from DBpedia URLs and retrieves corresponding Wikipedia pages
    """
    try:
        df['dbpedia_entities'] = df['dbpedia_entities'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    except Exception as e:
        print(f"Error parsing data: {e}")
        print("Please check the format of dbpedia_entities column")
        raise
    
    entities = []
    for entity_dic in tqdm(df['dbpedia_entities'], desc="Retrieving wikipedia pages"):
        for entity in entity_dic.values():
            entities.append(entity.split("/")[-1])  
    get_wikipedia_pages(entities=entities, sent_split=False, rerun=True, wiki_dir=wiki_dir)


def main():
    """
    Main function to run post-processing independently.
    Processes a specific dataset and saves the results.
    """
    dataset_name = os.path.join(PATHS["queries_dir"], "questions_20250507_100.csv")
    df = pd.read_csv(dataset_name)
    
    # Convert string representations of dictionaries to actual dictionaries
    df['dbpedia_entities'] = df['dbpedia_entities'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df['placeholders'] = df['placeholders'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    df_post_processed = post_process(df)
    print(df_post_processed.head())
    df_filtered = verify_and_filter_entities(df_post_processed)
    print(df_filtered.head())
    retrieve_wikipedia_pages(df_filtered) # retrieve the wikipedia pages
    df_filtered.to_csv(os.path.join(PATHS["queries_dir"], "questions_20250507_100_post_processed.csv"))

if __name__ == "__main__":
    main()
