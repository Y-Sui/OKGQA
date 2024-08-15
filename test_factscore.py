import os
import pandas as pd
import asyncio
from tqdm import tqdm
from dotenv import load_dotenv
from metrics import FactScoreLite
from methods.retriever import retrieve
from metrics.FactScoreLite import FactScore
from deepeval.test_case import LLMTestCase
from concurrent.futures import ProcessPoolExecutor


load_dotenv()


async def call_fact_score(
    generations: list[list],
    knowledge_sources: list[list[str]],
    rerun: bool = False,
):
    FactScoreLite.configs.model = "gpt-4o-mini"
    FactScoreLite.configs.facts_db_path = "OKG/wikipedia/.cache/facts.json"
    FactScoreLite.configs.decisions_db_path = "OKG/wikipedia/.cache/decisions.json"

    if rerun:
        os.remove(FactScoreLite.configs.facts_db_path)
        os.remove(FactScoreLite.configs.decisions_db_path)

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
    res_list = df_res["zero_shot"].to_list()

    # prepare data strcuture
    test_cases: list[LLMTestCase] = []
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

        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=res_list[index],
                retrieval_context=retrieved_context,
            )
        )

    # evaluate FactScore
    actual_output = [test_case.actual_output for test_case in test_cases]
    retrieval_context = [test_case.retrieval_context for test_case in test_cases]
    avg_fact_score, avg_init_fact_score_before_gamma = asyncio.run(
        call_fact_score(generations=actual_output, knowledge_sources=retrieval_context)
    )

    print(f"Average fact score: {avg_fact_score}")
    print(f"Average fact score before gamma: {avg_init_fact_score_before_gamma}")

    ...


if __name__ == "__main__":
    main()
