from pydantic import Field
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
from enum import Enum


class LLMTestCaseParams(Enum):
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"
    CONTEXT = "context"
    RETRIEVAL_CONTEXT = "retrieval_context"


@dataclass
class LLMTestCase:
    input: str
    actual_output: str
    expected_output: Optional[str] = None
    # the retrieval context can be a string or a list of strings, if not provided, it will be None
    retrieval_context: Optional[Union[str, List[str]]] = None
    additional_metadata: Optional[Dict] = None
    comments: Optional[str] = None
    _dataset_rank: Optional[int] = field(default=None, repr=False)
    _dataset_alias: Optional[str] = field(default=None, repr=False)
    _dataset_id: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        # Ensure `retrieval_context` is None or a list of strings
        if self.retrieval_context is not None:
            if (
                not isinstance(self.retrieval_context, list)
                and all(isinstance(item, str) for item in self.retrieval_context)
                and not isinstance(self.retrieval_context, str)
            ):
                print(type(self.retrieval_context))
                raise TypeError(
                    "'retrieval_context' must be None or a list of strings or string"
                )
