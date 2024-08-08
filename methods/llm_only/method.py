from typing import Union, Optional, Dict, Any, List
from ..models import GPTModel
from .few_shot.template import FewShotTemplate
from .zero_shot.template import ZeroShotTemplate
from .self_consistency.template import SelfConsistencyTemplate
from .cot.template import CoTTemplate


class LLMOnlyMethod:
    def __init__(
        self,
        model: Optional[Union[str, GPTModel]] = None,
        api_key: Optional[str] = None,
        async_mode: bool = True,
        verbose_mode: bool = False,
    ):
        self.model = (
            GPTModel(model=model, api_key=api_key) if isinstance(model, str) else model
        )
        self.generation_model = self.model.get_model_name()
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode

    def generate_zero_shot(self, query: str) -> str:
        prompt = ZeroShotTemplate.generate_answer(query)
        res = self.model.generate(prompt)

        return res[0]

    def generate_few_shot(self, query: str, shot_num: int = 3) -> str:
        pass

    def generate_self_consistency(
        self, query: str, shot_num: int = 3, run_num: int = 3
    ) -> str:
        pass

    def generate_cot(self, query: str) -> str:
        pass
