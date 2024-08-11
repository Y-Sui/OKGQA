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
from .template import HallucinationTemplate


class HallucinationMetric(BaseMetric):

    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, GPTModel]] = None,
        api_key: Optional[str] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = True,
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
        # if isinstance(test_case, ConversationalTestCase):
        #     test_case = validate_conversational_test_case(test_case, self)
        # check_llm_test_case_params(test_case, required_params, self)

        self.evaluation_cost = 0
        if self.async_mode:
            loop = get_or_create_event_loop()
            loop.run_until_complete(self.a_measure(test_case, _show_indicator=False))
        else:
            self.truths = self._generate_truths(test_case.retrieval_context)
            self.claims = self._generate_claims(test_case.actual_output)
            self.verdicts = self._generate_verdicts()
            self.score = self._calculate_score()
            self.reason = self._generate_reason()
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Truths:\n{prettify_list(self.truths)}",
                    f"Claims:\n{prettify_list(self.claims)}",
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    async def a_measure(
        self,
        test_case: Union[LLMTestCase],
        _show_indicator: bool = True,
    ) -> float:
        # if isinstance(test_case, ConversationalTestCase):
        #     test_case = validate_conversational_test_case(test_case, self)
        # check_llm_test_case_params(test_case, required_params, self)

        self.evaluation_cost = 0
        self.truths, self.claims = await asyncio.gather(
            self._a_generate_truths(test_case.retrieval_context),
            self._a_generate_claims(test_case.actual_output),
        )
        self.verdicts = await self._a_generate_verdicts()
        self.score = self._calculate_score()
        self.reason = await self._a_generate_reason()
        self.success = self.score >= self.threshold
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Truths:\n{prettify_list(self.truths)}",
                f"Claims:\n{prettify_list(self.claims)}",
                f"Verdicts:\n{prettify_list(self.verdicts)}",
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )

        return self.score

    async def _a_generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)

        prompt: dict = HallucinationTemplate.generate_reason(
            contradictions=contradictions,
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

    def _generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)

        prompt: dict = HallucinationTemplate.generate_reason(
            contradictions=contradictions,
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

    async def _a_generate_verdicts(self) -> List[HallucinationVerdict]:
        if len(self.claims) == 0:
            return []

        verdicts: List[HallucinationVerdict] = []
        prompt = HallucinationTemplate.generate_verdicts(
            claims=self.claims, retrieval_context="\n\n".join(self.truths)
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            verdicts = [HallucinationVerdict(**item) for item in data["verdicts"]]
            return verdicts
        else:
            try:
                res: Verdicts = await self.model.a_generate(prompt, schema=Verdicts)
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [HallucinationVerdict(**item) for item in data["verdicts"]]
                return verdicts

    def _generate_verdicts(self) -> List[HallucinationVerdict]:
        if len(self.claims) == 0:
            return []

        verdicts: List[HallucinationVerdict] = []
        prompt = HallucinationTemplate.generate_verdicts(
            claims=self.claims, retrieval_context="\n\n".join(self.truths)
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            verdicts = [HallucinationVerdict(**item) for item in data["verdicts"]]
            return verdicts
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts)
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [HallucinationVerdict(**item) for item in data["verdicts"]]
                return verdicts

    async def _a_generate_truths(
        self, retrieval_context: Union[str, list]
    ) -> List[str]:
        if isinstance(retrieval_context, list):
            prompt = HallucinationTemplate.generate_truths(
                text="\n\n".join(retrieval_context)
            )
        elif isinstance(retrieval_context, str):
            prompt = HallucinationTemplate.generate_truths(text=retrieval_context)
        if self.using_native_model:
            try:
                res, cost = await self.model.a_generate(prompt)
                self.evaluation_cost += cost
                data = trimAndLoadJson(res, self)
                return data["truths"]
            except:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["truths"]
        else:
            try:
                res: Truths = await self.model.a_generate(prompt, schema=Truths)
                return res.truths
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["truths"]

    def _generate_truths(self, retrieval_context: str) -> List[str]:
        prompt = HallucinationTemplate.generate_truths(
            text="\n\n".join(retrieval_context)
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["truths"]
        else:
            try:
                res: Truths = self.model.generate(prompt, schema=Truths)
                return res.truths
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["truths"]

    async def _a_generate_claims(self, actual_output: str) -> List[str]:
        prompt = HallucinationTemplate.generate_claims(text=actual_output)
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["claims"]
        else:
            try:
                res: Claims = await self.model.a_generate(prompt, schema=Claims)
                return res.claims
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["claims"]

    def _generate_claims(self, actual_output: str) -> List[str]:
        prompt = HallucinationTemplate.generate_claims(text=actual_output)
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["claims"]
        else:
            try:
                res: Claims = self.model.generate(prompt, schema=Claims)
                return res.claims
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["claims"]

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        faithfulness_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                faithfulness_count += 1

        score = faithfulness_count / number_of_verdicts
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
        return "Hallucination"
