import os
import pandas as pd
import numpy as np
import asyncio
import metrics
from tqdm import tqdm
from dotenv import load_dotenv
from methods.retriever import retrieve
from metrics.FactScoreLite import FactScore

load_dotenv()


async def call_fact_score(
    generations: list[list],
    knowledge_sources: list[list[str]],
    rerun: bool = False,
):
    metrics.FactScoreLite.configs.model = "gpt-4o-mini"
    metrics.FactScoreLite.configs.facts_db_path = "OKG/wikipedia/.cache/facts.json"
    metrics.FactScoreLite.configs.decisions_db_path = (
        "OKG/wikipedia/.cache/decisions.json"
    )

    if rerun:
        os.remove(metrics.FactScoreLite.configs.facts_db_path)
        os.remove(metrics.FactScoreLite.configs.decisions_db_path)

    fact_score = FactScore(gamma=10)

    scores, init_scores = await fact_score.get_factscore(
        generations=generations, knowledge_sources=knowledge_sources
    )

    return scores, init_scores


def main():
    # load dataset
    data = pd.read_csv("OKG/filtered_questions_50_v3.csv", index_col=0)
    data["dbpedia_entities"] = data["dbpedia_entities"].apply(lambda x: eval(x))
    data["dbpedia_entities_re"] = data["dbpedia_entities_re"].apply(lambda x: eval(x))

    # load model generation
    df_res = pd.read_csv("OKG/generation/50_v3.csv", index_col=0)
    schemas = df_res.keys().to_list()

    results = pd.DataFrame(columns=["method", "avg_fact_score", "avg_init_fact_score"])
    for schema in schemas:
        res_list = df_res[schema].to_list()

        # skip empty results
        if all(pd.isna(x) for x in res_list):
            continue

        # prepare data strcuture
        actual_outputs, retrieval_contexts = [], []
        for index, sample in data.iterrows():
            query = sample["question"]
            entities = []
            for entity in sample["dbpedia_entities"].values():
                entities.append(entity.split("/")[-1])

            retrieved_context: list[str] = []
            for entity in entities:
                for paragraph in retrieve(
                    model_name="bm25",
                    entity=entity,  # use entity name to index the wikipedia page
                    query=query,
                    top_k=5,
                    verbose=False,
                ):
                    retrieved_context.append(paragraph)

            actual_outputs.append(res_list[index])
            retrieval_contexts.append(retrieved_context)

        # evaluate FactScore
        avg_fact_score, avg_init_fact_score_before_gamma = asyncio.run(
            call_fact_score(
                generations=actual_outputs, knowledge_sources=retrieval_contexts
            )
        )

        print("--------------------------------------------------------")
        print(f"Average fact score of {schema}: {avg_fact_score}")
        print(
            f"Average fact score before gamma {schema}: {avg_init_fact_score_before_gamma}"
        )
        print("--------------------------------------------------------")

        results.loc[len(results)] = {
            "method": schema,
            "avg_fact_score": f"{avg_fact_score * 100:.3f}%",
            "avg_init_fact_score": f"{avg_init_fact_score_before_gamma * 100:.3f}%",
        }

    print(results)
    ...


if __name__ == "__main__":
    main()
