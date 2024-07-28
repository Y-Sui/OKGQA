from typing import Optional, Dict


class BaseMetric:
    score: Optional[float] = None
    score_breakdown: Dict = None
    reason: Optional[str] = None
    success: Optional[bool] = None
    evaluation_model: Optional[str] = None
    strict_mode: bool = False
    async_mode: bool = True
    verbose_mode: bool = True
    include_reason: bool = False
    error: Optional[str] = None
    evaluation_cost: Optional[float] = None
    verbose_logs: Optional[str] = None

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value

    @property
    def __name__(self):
        return "Base Metric"


class BaseConversationalMetric:
    score: Optional[float] = None
    score_breakdown: Dict = None
    reason: Optional[str] = None
    evaluation_model: Optional[str] = None
    error: Optional[str] = None
    # Not changeable for now
    strict_mode: bool = False
    async_mode: bool = False

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value

    @property
    def __name__(self):
        return "Base Conversational Metric"
