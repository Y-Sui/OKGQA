import logging
import openai
import os
from typing import Any, Dict, Optional, Sequence, Union, Tuple
from tenacity import retry, retry_if_exception_type, wait_exponential_jitter
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.schema import AIMessage, HumanMessage


def log_retry_error(retry_state):
    logging.error(
        f"OpenAI rate limit exceeded. Retrying: {retry_state.attempt_number} time(s)..."
    )


valid_gpt_models = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-4-32k",
    "gpt-4-0613",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0125",
]

valid_deepinfra_models = [
    "google/gemma-1.1-7b-it",
    "Qwen/Qwen2-7B-Instruct",
    "microsoft/WizardLM-2-7B",
    "meta/meta-llama-3-70b-instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2-72B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "microsoft/Phi-3-medium-4k-instruct",
]

default_gpt_model = "gpt-4o"
default_gpt_url = "https://api.openai.com/v1"
default_deepinfra_url = "https://api.deepinfra.com/v1/openai"


class GPTModel:
    def __init__(
        self,
        model: Optional[str] = None,
        _openai_api_key: Optional[str] = None,
        *args,
        **kwargs,
    ):
        self.model_name = None
        if isinstance(model, str):
            self.model_name = model
            if self.model_name not in valid_gpt_models:
                raise ValueError(
                    f"Invalid model. Available GPT models: {', '.join(model for model in valid_gpt_models)}"
                )
        elif model is None:
            self.model_name = default_gpt_model

        self._openai_api_key = _openai_api_key
        self.args = args
        self.kwargs = kwargs

    def load_model(self):
        return ChatOpenAI(
            model_name=self.model_name,
            openai_api_key=self._openai_api_key,
            *self.args,
            **self.kwargs,
        )

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    def generate(self, prompt: str) -> Tuple[str, float]:
        chat_model = self.load_model()
        with get_openai_callback() as cb:
            res = chat_model.invoke(prompt)
            return res.content, cb.total_cost

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    async def a_generate(self, prompt: str) -> Tuple[str, float]:
        chat_model = self.load_model()
        with get_openai_callback() as cb:
            res = await chat_model.ainvoke(prompt)
            return res.content, cb.total_cost

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    def generate_raw_response(self, prompt: str, **kwargs) -> Tuple[AIMessage, float]:
        if self.should_use_azure_openai():
            raise AttributeError

        chat_model = self.load_model().bind(**kwargs)
        with get_openai_callback() as cb:
            res = chat_model.invoke(prompt)
            return res, cb.total_cost

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    async def a_generate_raw_response(
        self, prompt: str, **kwargs
    ) -> Tuple[AIMessage, float]:
        if self.should_use_azure_openai():
            raise AttributeError

        chat_model = self.load_model().bind(**kwargs)
        with get_openai_callback() as cb:
            res = await chat_model.ainvoke(prompt)
        return res, cb.total_cost

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    def generate_samples(
        self, prompt: str, n: int, temperature: float
    ) -> Tuple[AIMessage, float]:
        chat_model = self.load_model()
        og_parameters = {"n": chat_model.n, "temp": chat_model.temperature}
        chat_model.n = n
        chat_model.temperature = temperature

        generations = chat_model._generate([HumanMessage(prompt)]).generations
        chat_model.temperature = og_parameters["temp"]
        chat_model.n = og_parameters["n"]

        completions = [r.text for r in generations]
        return completions

    def get_model_name(self):
        #   if self.should_use_azure_openai():
        #       return "azure openai"
        #   elif self.model_name:
        #       return self.model_name
        return self.model_name
