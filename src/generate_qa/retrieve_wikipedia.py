"""
This file is loaded by the post_process.py file, 
it is used to retrieve the wikipedia pages for the entities in the dataframe
"""
import wikipediaapi
import os
import pandas as pd
import nltk
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from multiprocessing import Pool      
from ..config.generate_qa_config import WIKI_CONFIG, PROCESSING_CONFIG

# if nltk is not installed, install it
if not nltk.data.find('tokenizers/punkt'):
    nltk.download('punkt')

def check_os_exists(entity: str, wiki_dir: str):
    """
    Check if a Wikipedia page for an entity already exists in the local storage.
    
    Args:
        entity (str): Name of the entity to check
        
    Returns:
        bool: True if the Wikipedia page exists locally, False otherwise
    """
    file_path = os.path.join(wiki_dir, f"{entity}.txt")
    return os.path.exists(file_path)

def fetch_wikipedia_page(entity: str, sent_split: bool = WIKI_CONFIG["sent_split"], rerun: bool = WIKI_CONFIG["rerun"], wiki_dir: str = ""):
    """
    Fetch and save a Wikipedia page for a given entity.
    
    Args:
        entity (str): Name of the entity to fetch Wikipedia page for
        sent_split (bool, optional): Whether to split the text into sentences. Defaults to WIKI_CONFIG["sent_split"]
        rerun (bool, optional): Whether to re-fetch existing pages. Defaults to WIKI_CONFIG["rerun"]
        
    Note:
        The page is saved in a text file with the entity name as filename.
        The content includes title, summary, and full text.
    """
    if check_os_exists(entity, wiki_dir) and not rerun:
        return
    
    wiki_wiki = wikipediaapi.Wikipedia(
        language=WIKI_CONFIG["language"],
        user_agent=WIKI_CONFIG["user_agent"]
    )
    page_py = wiki_wiki.page(entity)

    if not page_py.exists():
        return

    grd_context = f"<title>{page_py.title}</title>\n"
    grd_context += f"<summary>{page_py.summary}</summary>\n"
    grd_context += f"<text>{page_py.text}</text>\n"

    file_path = os.path.join(wiki_dir, f"{entity}.txt")
    
    if sent_split:
        sent_tokenize_list = sent_tokenize(grd_context)
        with open(file_path, "w", encoding='utf-8') as f:
            f.write("\n".join(sent_tokenize_list))
    else:
        with open(file_path, "w", encoding='utf-8') as f:
            f.write(grd_context)

def process_entity(args):
    """
    Process a single entity for Wikipedia page retrieval.
    
    Args:
        args (tuple): Tuple containing (entity, sent_split, rerun)
        
    Returns:
        bool: True if successful, False if an error occurred
    """
    entity, sent_split, rerun, wiki_dir = args
    try:
        fetch_wikipedia_page(entity, sent_split, rerun, wiki_dir)
        return True
    except Exception as e:
        print(f"Error fetching Wikipedia page for {entity}: {e}")
        return False

def get_wikipedia_pages(entities: list[str], sent_split: bool = WIKI_CONFIG["sent_split"], rerun: bool = WIKI_CONFIG["rerun"], wiki_dir: str = ""):
    """
    Retrieve Wikipedia pages for multiple entities in parallel.
    
    Args:
        entities (list[str]): List of entity names to fetch Wikipedia pages for
        sent_split (bool, optional): Whether to split text into sentences. Defaults to WIKI_CONFIG["sent_split"]
        rerun (bool, optional): Whether to re-fetch existing pages. Defaults to WIKI_CONFIG["rerun"]
        
    Note:
        Uses multiprocessing to fetch pages in parallel
        Removes duplicate entities while preserving order
    """
    # Remove duplicates while preserving order
    entities = list(dict.fromkeys(entities))
    
    # Prepare arguments for multiprocessing
    args = [(entity, sent_split, rerun, wiki_dir) for entity in entities]
    
    with Pool(PROCESSING_CONFIG["wiki_workers"]) as pool:
        list(tqdm(
            pool.imap(process_entity, args),
            total=len(args),
            desc="Fetching Wikipedia pages..."
        ))
