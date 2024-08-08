import os
import time
import random
import pandas as pd
from methods.llm_only.method import LLMOnlyMethod
from dotenv import load_dotenv

load_dotenv()


def load_dataset():
    queries = pd.read_csv("OKG/filtered_questions_50_v3.csv")


def main():
    generator = LLMOnlyMethod(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        async_mode=False,
        verbose_mode=True,
    )

    res = generator.generate_zero_shot("What is the capital of France?")
    print(res)


if __name__ == "__main__":
    main()
