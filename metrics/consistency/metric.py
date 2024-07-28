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
