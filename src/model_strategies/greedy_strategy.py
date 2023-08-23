from src.model_strategies.abstract import ModelStrategy
import numpy as np
from typing import Dict, Any


class GreedyStrategy(ModelStrategy):
    def __init__(self, estimation_model, cost_constraint, threshold=0.5):
        super().__init__(estimation_model)
        self.threshold = threshold
        self.cost_constraint = cost_constraint

    def _assign(self, batch: Dict[str, Any]):
        probs = batch["prob"]
        costs = batch["cost"]
        models = list(probs[0].keys())
        costs_chosen = []
        models_chosen = []
        total_cost = 0
        for prob, cost in zip(probs, costs):
            highest_prob = -1
            highest_prob_model = None
            for model in models:
                if prob[model] > highest_prob:
                    highest_prob_model = model
                    highest_prob = prob[model]
            if highest_prob_model is not None:
                if total_cost + cost[highest_prob_model] <= self.cost_constraint:
                    models_chosen.append(highest_prob_model)
                    costs_chosen.append(cost[highest_prob_model])
                    total_cost += cost[highest_prob_model]
                else:
                    models_chosen.append(None)
                    costs_chosen.append(None)
        return models_chosen, costs_chosen
