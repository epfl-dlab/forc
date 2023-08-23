import numpy as np
from typing import Dict, Any

from src.model_strategies.abstract import ModelStrategy


class PerformanceMaximizingStrategy(ModelStrategy):
    def __init__(self, estimation_model):
        super().__init__(estimation_model)

    def _assign(self, batch: Dict[str, Any]):
        probs = batch["prob"]
        costs = batch["cost"]
        models = list(probs[0].keys())
        costs_chosen = []
        models_chosen = []

        for prob, cost in zip(probs, costs):
            highest_prob = np.NINF
            chosen_model = None
            chosen_cost = None
            for model in models:
                if prob[model] > highest_prob:
                    highest_prob = prob[model]
                    chosen_model = model
                    chosen_cost = cost[model]
            models_chosen.append(chosen_model)
            costs_chosen.append(chosen_cost)
        return models_chosen, costs_chosen
