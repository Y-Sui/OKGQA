import pandas as pd
import os
import requests
import asyncio
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .retrieve_wikipedia import get_wikipedia_pages

    
def check_url(url: str) -> bool:
    """
    Specialized checker for DBpedia resources with enhanced headers and SSL handling
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive'
    }

    try:
        # First try HEAD request
        response = requests.head(
            url.replace("http://", "https://"),
            headers=headers,
            timeout=15,
            allow_redirects=True,
            verify=True  # DBpedia uses valid SSL
        )
        
        # Fallback to GET if HEAD is not allowed
        if response.status_code == 405:
            response = requests.get(
                url.replace("http://", "https://"),
                headers=headers,
                timeout=15,
                allow_redirects=True,
                verify=True,
                stream=True  # Don't download content
            )
            
        return response.ok

    except requests.exceptions.SSLError:
        return False
    except requests.exceptions.RequestException:
        return False


def post_process(df: pd.DataFrame):
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
    valid_rows = set()
    print(f"Total rows to process: {len(df)}")

    # use ThreadPoolExecutor for parallel calling of check_url()
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
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


def retrieve_wikipedia_pages(df: pd.DataFrame):
    """
    retrieve the wikipedia pages for the entities in the dataframe
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
    get_wikipedia_pages(entities=entities, sent_split=False, rerun=True)


def main():
    dir_path = "/mnt/250T_ceph/tristanysui/okgqa"
    dataset_name = dir_path + f"/questions_20250506_200651.csv"
    df = pd.read_csv(dataset_name)
    
    # Convert string representations of dictionaries to actual dictionaries
    df['dbpedia_entities'] = df['dbpedia_entities'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df['placeholders'] = df['placeholders'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    df_post_processed = post_process(df)
    print(df_post_processed.head())
    df_filtered = verify_and_filter_entities(df_post_processed)
    print(df_filtered.head())
    retrieve_wikipedia_pages(df_filtered) # retrieve the wikipedia pages
    df_filtered.to_csv(dir_path + f"/questions_20250506_200651_post_processed.csv")

if __name__ == "__main__":
    main()
