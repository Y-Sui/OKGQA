import os
from metrics.hallucination.metric import HallucinationMetric
from metrics.llm_test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()


test_case = LLMTestCase(
    # input="What if these shoes don't fit?",
    # input="xxx",
    input="",
    actual_output="We offer a 30-day full refund at no extra cost. We have a pair of shoes",  # extract claims
    expected_output="Paris",
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost."
        # "We offer a 30-day full refund at no extra cost.",
        # "((custormer, 30-day full refund, no extra cost))",
    ],  # extract facts
)

metric = HallucinationMetric(
    threshold=0.5,
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    include_reason=True,
    strict_mode=False,
)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
