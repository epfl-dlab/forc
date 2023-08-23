from abc import ABC
from typing import List, Dict, Any
import tiktoken
import math


class ModelStrategy(ABC):
    def __init__(self, estimation_model=None):
        self.estimation_model = estimation_model
        if estimation_model is not None:
            self.openai_models = self.estimation_model.models_info
        else:
            self.openai_models = None

    def _estimate_cost(self, batch: List[str]):
        costs = []
        for sample in batch:
            sample_cost = {}
            for openai_model in self.openai_models:
                model_name = openai_model.model_name
                model_enc = tiktoken.encoding_for_model(model_name)
                model_cost = openai_model.token_cost

                # prompt cost # todo leave as prompt cost or just do input query?
                prompt_cost = len(model_enc.encode(sample)) * model_cost / 1000.

                # estimated completion cost
                cost = prompt_cost + openai_model.average_output_len * model_cost / 1000

                sample_cost[model_name] = cost
            costs.append(sample_cost)
        return costs

    def _estimate_success_probs(self, batch: List[str], batch_size: int):
        model_probs = []
        num_iters = math.ceil(len(batch) / batch_size)
        for i in range(num_iters):
            curr_batch = batch[i * batch_size:(i + 1) * batch_size]
            model_probs.extend(self.estimation_model.test_batch(curr_batch))
        return model_probs

    def _assign(self, batch: Dict[str, Any]):
        raise NotImplementedError

    def __call__(self, batch: List[str], batch_size: int = 8, return_total_cost: bool = False):
        if type(batch) == str:
            batch = [batch]

        model_costs = self._estimate_cost(batch)
        if len(self.openai_models) == 1:
            model_name = self.openai_models[0].model_name
            model_assignments = [model_name] * len(batch)
            assigned_costs = [m_c[model_name] for m_c in model_costs]
        else:
            model_probs = self._estimate_success_probs(batch, batch_size)
            model_assignments, assigned_costs = self._assign({'prob': model_probs, 'cost': model_costs})

        if return_total_cost:
            total_cost = assigned_costs.sum()
            return model_assignments, assigned_costs, total_cost
        else:
            return model_assignments, assigned_costs
