import os
import pytest
import time
import pandas as pd
import deepeval
import subprocess
from tqdm import tqdm
from methods.retriever import retrieve
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import *
from deepeval.models import DeepEvalBaseLLM
from typing import Union, Optional

load_dotenv()

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
            expected_output="",
        )
    )


# use GEval to define the evaluation criteria and run the evaluation
empowerment_metric = GEval(
    name="Empowerment",
    criteria="""
    Evaluate whether the 'Actual Output' can help the reader understand the topic and make informed decisions regarding the 'Input'. 
    A response with high empowerment provides accurate information and explanations that enhance the reader’s understanding.\
    When evaluating empowerment, consider the relevance of the information provided in the 'Actual Output' to the 'Input' and the 'Retrieval Context'.\
    """,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model="gpt-4o-mini",
)

comprenhensiveness_metric = GEval(
    name="Comprehensiveness",
    criteria="""
    Evalute the extent to which the 'Actual Output' covers all aspects and details of the question 'Input'. 
    A comprehensive answer should thoroughly address every part of the question, leaving no important points unaddressed.
    When evaluating comprehensiveness, consider the relevance of the information provided in the 'Actual Output' to the 'Input' and the 'Retrieval Context'.
    """,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model="gpt-4o-mini",
)

correctness_metric = GEval(
    name="Correctness",
    criteria="""
    Measure how clearly and specifically the 'Actual output' responds to the question 'input'. 
    A highly direct response stays focused on the question, providing clear and unambiguous information
    When evaluating correctness, consider the relevance of the information provided in the 'Actual Output' to the 'Input' and the 'Retrieval Context'.
    """,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model="gpt-4o-mini",
)


@pytest.mark.parametrize("test_case", EvaluationDataset(test_cases))
def test_deepeval_score(test_case: LLMTestCase):
    answer_relevancy_metric = AnswerRelevancyMetric(
        model="gpt-4o-mini",
        threshold=0.001,
        strict_mode=False,
    )  # determines whether the LLM outputs are relevant to the input.
    context_relevancy_metric = ContextualRelevancyMetric(
        model="gpt-4o-mini",
        threshold=0.001,
        strict_mode=False,
    )  # determines whether the LLM outputs are relevant to the retrieval context.
    bias_metric = BiasMetric(
        model="gpt-4o-mini",
        strict_mode=False,
    )  # determines whether the LLM outputs contains gender, racial, or political bias.
    faithfulness_metric = FaithfulnessMetric(strict_mode=False)
    toxicity_metric = ToxicityMetric(
        model="gpt-4o-mini",
        strict_mode=False,
    )  # determines whether the LLM outputs are toxic.
    assert_test(
        test_case,
        [
            answer_relevancy_metric,
            context_relevancy_metric,
            bias_metric,
            toxicity_metric,
            faithfulness_metric,
            empowerment_metric,
            comprenhensiveness_metric,
            correctness_metric,
        ],
    )


@deepeval.on_test_run_end
def function_to_be_called_after_test_run():
    print("Test finished!")


def run_command():
    command = ["deepeval", "test", "run", "test_deepeval.py", "-c", "-i", "-n", "32"]
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command)

    if result.returncode != 0:
        raise ValueError("Test failed!")
    else:
        print("Test passed!")


def main():
    run_command()


if __name__ == "__main__":
    main()
