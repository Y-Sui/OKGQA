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
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import q_prefix, q_example

# set the maximum number of retries
MAX_RETRIES = 10

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

uuid = str(uuid.uuid4()).split("-")[-1]

# set the dataset name 
dataset_name = f"questions_{uuid}.csv"

def call_gpt(gpt_prompt, examples=""):
    attempts = 0
    while attempts < MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant designed to output JSON.",
                    },
                    (
                        {
                            "role": "user",
                            "content": gpt_prompt + "\nExamples: " + examples,
                        }
                        if examples
                        else {"role": "user", "content": gpt_prompt}
                    ),
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling GPT: {e}")
            continue


def process_call(i):
    while True:
        try:
            raw_output = call_gpt(q_prefix, q_example)
            output = ast.literal_eval(raw_output)
            content = output.get("questions", "N/A")
            
            if content != "N/A":
                return content
        
        except (SyntaxError, ValueError):
            continue
        
def post_process(df: pd.DataFrame):
    unique_sample = set()
    rows_to_keep = []

    for i, row in df.iterrows():
        type = row["type"]
        question = row["question"]
        placeholders = row["placeholders"]
        dbpedia_entities = row["dbpedia_entities"]
        try:
            # frozenset is hashable
            placeholders = frozenset(row["placeholders"].items())
            dbpedia_entities = frozenset(row["dbpedia_entities"].items())
            if (row["type"], placeholders, dbpedia_entities) not in unique_sample:
                unique_sample.add((row["type"], placeholders, dbpedia_entities))
                rows_to_keep.append(i)

        except Exception as e:
            print(f"Error processing row {i}: {e}")
            print("placeholders: ", placeholders)
            print((row["type"], placeholders, dbpedia_entities))
            print("question: ", question)
            pass

    print(f"Number of unique samples: {len(unique_sample)}")
    df_filtered = df.iloc[rows_to_keep].reset_index(drop=True)

    return df_filtered

def check_url(url):
    try:
        response = requests.get(url, timeout=5)  # Added a timeout for faster failure
        return response.status_code == 200
    except:
        return False


def verify_and_filter_entities(df):
    valid_rows = set()

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {}
        for index, row in df.iterrows():
            for key, url in row['dbpedia_entities'].items():
                future = executor.submit(check_url, url)
                futures[future] = (index, key)

        url_results = {}  # store the results of the futures
        # process the futures as they are completed
        for future in tqdm(as_completed(futures), total=len(futures)):
            url_results[futures[future]] = future.result()

    # iterate through the DataFrame rows to filter out invalid ones
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        valid = True
        for key, url in row['dbpedia_entities'].items():
            # if any URL is invalid, set valid to False and break
            if not url_results[(index, key)]:
                valid = False
                break
        if valid:
            valid_rows.add(index)

    filtered_df = df.loc[list(valid_rows)].reset_index(drop=True)
    return filtered_df

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def main():

    if os.path.exists(dataset_name):
        df = pd.read_csv(dataset_name, index_col=0)
        df["dbpedia_entities"] = df["dbpedia_entities"].apply(lambda x: eval(x))
        df["placeholders"] = df["placeholders"].apply(lambda x: eval(x))
    else:   
        results = []
        batch_size = 300
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_call, i) for i in range(batch_size)]
            for future in tqdm(as_completed(futures), total=batch_size):
                result = future.result()
                if result:
                    results.append(result)

        # Convert results to a DataFramex
        df = pd.DataFrame([item for sublist in results for item in sublist])
        df.to_csv(dataset_name)
        
    df_post_processed = post_process(df)
    filtered_df = verify_and_filter_entities(df_post_processed)
    filtered_df.to_csv(f"filtered_{dataset_name}.csv")

    # calculate the token length of the questions, the number of questions, the number of unique dbpedia entities
    number_of_questions = len(filtered_df)

    # Count the number of unique DBpedia entities
    unique_dbpedia_entities = set()
    for placeholder in filtered_df["placeholders"]:
        unique_dbpedia_entities.update(placeholder.values())
        
    print(unique_dbpedia_entities)

    # Display results
    print(f"Number of questions: {number_of_questions}")
    print(f"Number of unique DBpedia entities: {len(unique_dbpedia_entities)}")

    avg_token_question = 0
    for question in filtered_df["question"]:
        num_token = num_tokens_from_messages([{"content": question}])
        avg_token_question += num_token

    avg_token_question = avg_token_question / number_of_questions
    print(f"Average token length of questions: {avg_token_question}")

    type_counts = filtered_df["type"].value_counts()
    print(type_counts)

    type_naturalness_counts = filtered_df.groupby(['type', 'naturalness']).size().unstack()
    print(type_naturalness_counts)

    type_difficulty_counts = filtered_df.groupby(['type', 'difficulty']).size().unstack()
    print(type_difficulty_counts)
    
    
if __name__ == "__main__":
    main()