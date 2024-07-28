import asyncio
import os
from typing import List, Optional, Union
from ..base_metric import BaseMetric
from ..utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    prettify_list,
    get_or_create_event_loop,
)
from ..llm_test_case import LLMTestCase
from ..models import GPTModel
from .schema import *
from .template import ContextualRelevancyTemplate


class ContextualRelevancyMetric(BaseMetric):

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, GPTModel]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model = (
            GPTModel(model=model, _openai_api_key=api_key)
            if isinstance(model, str)
            else model
        )
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.using_native_model = True

    def measure(self, test_case: LLMTestCase) -> float:

        self.evaluation_cost = 0
        if self.async_mode:
            loop = get_or_create_event_loop()
            loop.run_until_complete(self.a_measure(test_case, _show_indicator=False))
        else:
            self.verdicts: List[ContextualRelevancyVerdict] = self._generate_verdicts(
                test_case.input, test_case.retrieval_context
            )
            self.score = self._calculate_score()
            self.reason = self._generate_reason(test_case.input)
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
    ) -> float:

        self.evaluation_cost = 0 if self.using_native_model else None

        self.verdicts: List[ContextualRelevancyVerdict] = (
            await self._a_generate_verdicts(
                test_case.input, test_case.retrieval_context
            )
        )
        self.score = self._calculate_score()
        self.reason = await self._a_generate_reason(test_case.input)
        self.success = self.score >= self.threshold
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Verdicts:\n{prettify_list(self.verdicts)}",
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )

        return self.score

    async def _a_generate_reason(self, input: str):
        if self.include_reason is False:
            return None

        irrelevancies = []
        for verdict in self.verdicts:
            if verdict.verdict.lower() == "no":
                irrelevancies.append(verdict.reason)

        prompt: dict = ContextualRelevancyTemplate.generate_reason(
            input=input,
            irrelevancies=irrelevancies,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = await self.model.a_generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_reason(self, input: str):
        if self.include_reason is False:
            return None

        irrelevancies = []
        for verdict in self.verdicts:
            if verdict.verdict.lower() == "no":
                irrelevancies.append(verdict.reason)

        prompt: dict = ContextualRelevancyTemplate.generate_reason(
            input=input,
            irrelevancies=irrelevancies,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = self.model.generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    async def _a_generate_verdict(self, prompt: str) -> ContextualRelevancyVerdict:
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return ContextualRelevancyVerdict(**data)
        else:
            try:
                res = await self.model.a_generate(
                    prompt, schema=ContextualRelevancyVerdict
                )
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return ContextualRelevancyVerdict(**data)

    async def _a_generate_verdicts(
        self, text: str, retrieval_context: List[str]
    ) -> ContextualRelevancyVerdict:
        tasks = []
        for context in retrieval_context:
            prompt = ContextualRelevancyTemplate.generate_verdict(
                text=text, context=context
            )
            task = asyncio.create_task(self._a_generate_verdict(prompt))
            tasks.append(task)
        verdicts = await asyncio.gather(*tasks)
        return verdicts

    def _generate_verdicts(
        self, text: str, retrieval_context: List[str]
    ) -> List[ContextualRelevancyVerdict]:
        verdicts: List[ContextualRelevancyVerdict] = []
        for context in retrieval_context:
            prompt = ContextualRelevancyTemplate.generate_verdict(
                text=text, context=context
            )
            if self.using_native_model:
                res, cost = self.model.generate(prompt)
                self.evaluation_cost += cost
                data = trimAndLoadJson(res, self)
                verdict = ContextualRelevancyVerdict(**data)
            else:
                try:
                    res = self.model.generate(prompt, schema=ContextualRelevancyVerdict)
                    verdict = res
                except TypeError:
                    res = self.model.generate(prompt)
                    data = trimAndLoadJson(res, self)
                    verdict = ContextualRelevancyVerdict(**data)

            verdicts.append(verdict)

        return verdicts

    def _calculate_score(self):
        total_verdicts = len(self.verdicts)
        if total_verdicts == 0:
            return 0

        relevant_nodes = 0
        for verdict in self.verdicts:
            if verdict.verdict.lower() == "yes":
                relevant_nodes += 1

        score = relevant_nodes / total_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Contextual Relevancy"
