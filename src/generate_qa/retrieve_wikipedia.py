"""
This file is loaded by the post_process.py file, 
it is used to retrieve the wikipedia pages for the entities in the dataframe
"""
import wikipediaapi
import os
import pandas as pd
import nltk
from dotenv import load_dotenv
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from multiprocessing import Pool, cpu_count
from functools import partial

load_dotenv()
nltk.download('punkt')

WIKI_DIR = "/mnt/250T_ceph/tristanysui/okgqa/wikipedia"
os.makedirs(WIKI_DIR, exist_ok=True)

def check_os_exists(entity: str):
    file_path = os.path.join(WIKI_DIR, f"{entity}.txt")
    return os.path.exists(file_path)

def fetch_wikipedia_page(entity: str, sent_split: bool = True, rerun: bool = False):
    if check_os_exists(entity) and not rerun:
        return
    
    user_agent = "OpenKGQA/0.0 (yuansui08@gmail.com)"
    wiki_wiki = wikipediaapi.Wikipedia(language="en", user_agent=user_agent)
    page_py = wiki_wiki.page(entity)

    if not page_py.exists():
        return

    grd_context = f"<title>{page_py.title}</title>\n"
    grd_context += f"<summary>{page_py.summary}</summary>\n"
    grd_context += f"<text>{page_py.text}</text>\n"

    file_path = os.path.join(WIKI_DIR, f"{entity}.txt")
    
    if sent_split:
        sent_tokenize_list = sent_tokenize(grd_context)
        with open(file_path, "w", encoding='utf-8') as f:
            f.write("\n".join(sent_tokenize_list))
    else:
        with open(file_path, "w", encoding='utf-8') as f:
            f.write(grd_context)

def process_entity(args):
    entity, sent_split, rerun = args
    try:
        fetch_wikipedia_page(entity, sent_split, rerun)
        return True
    except Exception as e:
        print(f"Error fetching Wikipedia page for {entity}: {e}")
        return False

def get_wikipedia_pages(entities: list[str], sent_split: bool, rerun: bool):
    # Remove duplicates while preserving order
    entities = list(dict.fromkeys(entities))
    
    # Prepare arguments for multiprocessing
    args = [(entity, sent_split, rerun) for entity in entities]
    
    # Use number of CPU cores minus 1 to leave one core free
    num_cores = max(1, cpu_count() - 1)
    
    with Pool(num_cores) as pool:
        list(tqdm(
            pool.imap(process_entity, args),
            total=len(args),
            desc="Fetching Wikipedia pages..."
        ))

def main():
    data = pd.read_csv("/mnt/250T_ceph/tristanysui/okgqa/queries/questions_20250507_100.csv", index_col=0)
    data["dbpedia_entities"] = data["dbpedia_entities"].apply(lambda x: eval(x))

    entities = []
    for entity_dic in data["dbpedia_entities"]:
        for entity in entity_dic.values():
            entities.append(entity.split("/")[-1])

    get_wikipedia_pages(entities=entities, sent_split=False, rerun=True)

if __name__ == "__main__":
    main()
