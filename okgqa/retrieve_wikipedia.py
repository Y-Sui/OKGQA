import wikipediaapi
import asyncio
import aiofiles
import nest_asyncio
import os
import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from nltk.tokenize import sent_tokenize

load_dotenv()
nest_asyncio.apply()

import nltk

nltk.download('punkt')


def check_os_exists(entity: str):
    os.makedirs("OKG/wikipedia", exist_ok=True)
    file_path = f"OKG/wikipedia/{entity}.txt"

    if os.path.exists(file_path):
        print(f"File {file_path} already exists, skipping.")
        return True
    return False


async def fetch_wikipedia_page(
    entity: str, sent_split: bool = True, rerun: bool = False
):
    if check_os_exists(entity) and not rerun:
        return
    user_agent = "OpenKGQA/0.0 ("")"
    wiki_wiki = wikipediaapi.Wikipedia(language="en", user_agent=user_agent)
    page_py = wiki_wiki.page(entity)

    grd_context = ""
    if page_py.exists():
        grd_context += "Page - Title: %s\n" % page_py.title
        grd_context += "Page - Summary: %s" % page_py.summary
        grd_context += "Page - Text: %s" % page_py.text

        if sent_split:
            sent_tokenize_list = sent_tokenize(grd_context)

            async with aiofiles.open(f"OKG/wikipedia/{entity}.txt", "w") as f:
                for sentence in sent_tokenize_list:
                    await f.write(sentence + "\n")
        else:
            async with aiofiles.open(f"OKG/wikipedia/{entity}.txt", "w") as f:
                await f.write(grd_context)
    else:
        print(f"Page {entity} does not exist")


async def get_wikipedia_pages(entities: list[str], sent_split: bool, rerun: bool):
    tasks = [fetch_wikipedia_page(entity, sent_split, rerun) for entity in entities]
    progress_bar = tqdm_asyncio(total=len(tasks), desc="Fetching Wikipedia pages...")
    for task in asyncio.as_completed(tasks):
        await task
        progress_bar.update(1)
    progress_bar.close()


def main():
    data = pd.read_csv("OKG/filtered_questions_50_v3.csv", index_col=0)
    data["dbpedia_entities"] = data["dbpedia_entities"].apply(lambda x: eval(x))

    entities = []
    for entity_dic in data["dbpedia_entities"]:
        for entity in entity_dic.values():
            entities.append(entity.split("/")[-1])

    asyncio.run(get_wikipedia_pages(entities=entities, sent_split=False, rerun=True))


if __name__ == "__main__":
    main()
