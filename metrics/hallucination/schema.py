from typing import List
from pydantic import BaseModel, Field


class HallucinationVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[HallucinationVerdict]


class Truths(BaseModel):
    truths: List[str]


class Claims(BaseModel):
    claims: List[str]


class Reason(BaseModel):
    reason: str
