from src.model_strategies.abstract import ModelStrategy
import numpy as np
from typing import Dict, Any


class ThresholdingStrategy(ModelStrategy):
    def __init__(self, estimation_model, all_wrong_strategy='biggest', multiple_correct_strategy='cheapest', threshold=0.5):
        super().__init__(estimation_model)
        self.threshold = threshold
        self.all_wrong_strategy = all_wrong_strategy
        self.multiple_correct_strategy = multiple_correct_strategy

    def _assign(self, batch: Dict[str, Any]):
        probs = batch["prob"]
        costs = batch["cost"]
        models = list(probs[0].keys())
        costs_chosen = []
        models_chosen = []
        for prob, cost in zip(probs, costs):
            cheapest_model = None
            biggest_model = None
            min_cost = np.inf
            max_cost = np.NINF
            models_solving = []
            probs_solving = []
            costs_solving = []
            for model in models:
                if cost[model] < min_cost:
                    min_cost = cost[model]
                    cheapest_model = model
                if cost[model] > max_cost:
                    max_cost = cost[model]
                    biggest_model = model
                if prob[model] > self.threshold:
                    models_solving.append(model)
                    probs_solving.append(prob[model])
                    costs_solving.append(cost[model])
            if len(models_solving) == 0:
                if self.all_wrong_strategy == "biggest":
                    models_chosen.append(biggest_model)
                    costs_chosen.append(max_cost)
                elif self.all_wrong_strategy == "cheapest":
                    models_chosen.append(cheapest_model)
                    costs_chosen.append(min_cost)
                elif self.all_wrong_strategy in models:
                    models_chosen.append(self.all_wrong_strategy)
                    costs_chosen.append(cost[self.all_wrong_strategy])
                else:
                    raise NotImplementedError
            elif len(models_solving) == 1:
                models_chosen.append(models_solving[0])
                costs_chosen.append(costs_solving[0])
            else:
                sorted_data = sorted(zip(models_solving, probs_solving, costs_solving), key=lambda x: x[2])
                models_solving, probs_solving, costs_solving = zip(*sorted_data)
                if self.multiple_correct_strategy == "cheapest":
                    models_chosen.append(models_solving[0])
                    costs_chosen.append(costs_solving[0])
                elif self.multiple_correct_strategy == "biggest":
                    models_chosen.append(models_solving[-1])
                    costs_chosen.append(costs_solving[-1])
                else:
                    raise NotImplementedError
        return models_chosen, costs_chosen






