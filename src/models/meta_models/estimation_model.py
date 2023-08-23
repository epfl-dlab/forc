from dataclasses import dataclass
from typing import List, Any, Dict

import numpy as np


@dataclass
class OpenAIModelInfo:
    model_name: str
    token_cost: float
    average_output_len: int
    model_prefix: str


class EstimationModel:
    def __init__(self, models_info: List[OpenAIModelInfo]):
        self.models_info = models_info

    def test_batch(self, batch: List[str]):
        raise NotImplementedError
