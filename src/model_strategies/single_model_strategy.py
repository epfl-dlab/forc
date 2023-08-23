from src.model_strategies.abstract import ModelStrategy
from typing import List, Dict, Any


class SingleModelStrategy(ModelStrategy):
    def __init__(self, model):
        super().__init__(None)
        self.model = model

    def _assign(self, batch: Dict[str, Any]):
        pass

    def __call__(self, batch: List[str], batch_size: int = 8, return_total_cost: bool = False):
        model_costs = self._estimate_cost(batch)
        costs = [sample[self.model] for sample in model_costs]
        return [self.model] * len(batch), costs
