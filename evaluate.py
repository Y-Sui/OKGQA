import os
import pandas as pd
import FactScoreLite
from tqdm import tqdm
from dotenv import load_dotenv
from metrics.hallucination.metric import HallucinationMetric
from metrics.llm_test_case import LLMTestCase
from methods.retriever import retrieve
from FactScoreLite import FactScore

load_dotenv()


def call_local_metric(
    test_cases: list[LLMTestCase],
):
    # evaluate metric
    metric = HallucinationMetric(
        threshold=0.5,
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        include_reason=True,
        strict_mode=False,
        async_mode=True,
    )

    avg_facutal_score = 0.0
    for idx, test_case in tqdm(enumerate(test_cases)):
        metric.measure(test_case)
        avg_facutal_score += metric.score

        if idx < 5:
            print(f"Score: {metric.score}")
            print(f"Reason: {metric.reason}")

    avg_facutal_score /= len(test_cases)
    print(f"Average factual score: {avg_facutal_score}")

    return avg_facutal_score


def call_fact_score(
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

    scores, init_scores = FactScore(gamma=10).get_factscore(
        generations=generations,
        knowledge_sources=knowledge_sources,
    )
    print(f"Average fact score: {scores}")

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
    # call local metric
    avg_facutal_score = call_local_metric(test_cases)

    # evaluate FactScore
    actual_output = [test_case.actual_output for test_case in test_cases[:5]]
    retrieval_context = [test_case.retrieval_context for test_case in test_cases[:5]]
    avg_fact_score, _ = call_fact_score(
        generations=actual_output, knowledge_sources=retrieval_context
    )

    ...


if __name__ == "__main__":
    main()
