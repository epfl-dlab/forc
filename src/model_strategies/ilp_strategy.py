from src.model_strategies.abstract import ModelStrategy
import numpy as np
import pulp as pl
from src.utils.general_helpers import dict_list_to_matrix, modify_zero_vals
from typing import Dict, Any


class ILPStrategy(ModelStrategy):
    def __init__(
            self,
            estimation_model,
            cost_constraint=None,
            quality_constraint=None,
            alpha=0.5,
            threshold=0.5,
            minimise_cost=False,
            maximise_performance=False,
            maximise_budget=False,
            time_limit=None,
            nonsolved_penalty=0,
            model_prob=None,
            binarize=False,
    ):
        super().__init__(estimation_model)
        self.cost_constraint = cost_constraint
        self.quality_constraint = quality_constraint
        self.alpha = alpha
        self.threshold = threshold
        self.minimise_cost = minimise_cost
        self.maximise_performance = maximise_performance
        self.maximise_budget = maximise_budget
        self.time_limit = time_limit
        self.nonsolved_penalty = nonsolved_penalty
        self.model_prob = model_prob
        self.binarize = binarize
        self._update_obj_func_params()

    def _update_obj_func_params(self):
        if self.cost_constraint is None and self.quality_constraint is None:
            return
        if self.cost_constraint is not None and self.quality_constraint is None:
            self.maximise_performance = True
        if self.quality_constraint is not None and self.cost_constraint is None:
            self.minimise_cost = True

    def resolve_ILP(self, problem, c, v, models):
        models_chosen = [None]*len(v)
        costs_chosen = [None]*len(c)
        for var in problem.variables():
            if var.name.startswith("x") and var.value() == 1:
                _, idx_i, idx_j = var.name.split("_")
                idx_i, idx_j = int(idx_i), int(idx_j)
                models_chosen[idx_i] = models[idx_j]
                costs_chosen[idx_i] = c[idx_i][idx_j]
        return models_chosen, costs_chosen

    def solve_ILP(self, probs, models, costs, model_prob):
        c = dict_list_to_matrix(costs)
        v = dict_list_to_matrix(probs)
        if self.binarize:
            v = np.where(v > self.threshold, 1, 0)
        if model_prob is not None:
            v = modify_zero_vals(v, model_prob, self.threshold)

        # defining the problem
        ilp_problem = pl.LpProblem("ILP_problem", pl.LpMaximize)

        # adding variables
        x = {}
        y = {}
        for i in range(len(probs)):
            for j in range(len(models)):
                x[(i, j)] = pl.LpVariable(f"x_{i}_{j}", cat='Binary')

        for i in range(len(probs)):
            y[i] = pl.LpVariable(f"y_{i}", cat='Binary')

        # adding objective
        if self.cost_constraint is None and self.quality_constraint is None:
            # unconstrained problem
            objective = pl.lpSum(
                self.alpha * v[i][j] * x[(i, j)] - (1 - self.alpha) * c[i][j] * x[(i, j)] for i in range(len(probs)) for
                j in range(len(models))) - pl.lpSum(self.nonsolved_penalty * y[i] for i in range(len(probs)))
        elif self.maximise_performance:
            if self.minimise_cost:
                objective = pl.lpSum(
                    self.alpha * v[i][j] * x[(i, j)] - (1 - self.alpha) * c[i][j] * x[(i, j)] for i in range(len(probs))
                    for j in
                    range(len(models))) - pl.lpSum(
                    self.nonsolved_penalty * y[i] for i in range(len(probs)))
            elif self.maximise_budget:
                objective = pl.lpSum(
                    self.alpha * v[i][j] * x[(i, j)] + (1 - self.alpha) * c[i][j] * x[(i, j)] for i in range(len(probs))
                    for j in
                    range(len(models))) - pl.lpSum(
                    self.nonsolved_penalty * y[i] for i in range(len(probs)))
            else:
                objective = pl.lpSum(v[i][j] * x[(i, j)] for i in range(len(probs)) for j in
                                     range(len(models))) - pl.lpSum(
                    self.nonsolved_penalty * y[i] for i in range(len(probs)))
        elif self.minimise_cost:
            objective = pl.lpSum(- c[i][j] * x[(i, j)] for i in range(len(probs)) for j in
                                 range(len(models))) - pl.lpSum(
                self.nonsolved_penalty * y[i] for i in range(len(probs)))
        elif self.maximise_budget:
            objective = pl.lpSum(c[i][j] * x[(i, j)] for i in range(len(probs)) for j in
                                 range(len(models))) - pl.lpSum(
                self.nonsolved_penalty * y[i] for i in range(len(probs)))
        else:
            raise NotImplementedError

        ilp_problem += objective

        # adding constraint
        for i in range(len(probs)):
            ilp_problem += pl.lpSum(x[(i, j)] for j in range(len(models))) <= 1

        for i in range(len(probs)):
            ilp_problem += pl.lpSum(x[(i, j)] for j in range(len(models))) + y[i] == 1

        if self.cost_constraint is not None:
            ilp_problem += pl.lpSum(
                c[i][j] * x[(i, j)] for i in range(len(probs)) for j in range(len(models))) <= self.cost_constraint

        if self.quality_constraint is not None:
            # adding constraint
            if self.quality_constraint < 1:
                # constraint specified as fraction of samples needed to be solved
                quality_constraint = round(self.quality_constraint * len(probs))
            else:
                # constraint specified as number of samples needed to be solved
                quality_constraint = self.quality_constraint

            ilp_problem += pl.lpSum(
                v[i][j] * x[(i, j)] for i in range(len(probs)) for j in range(len(models))) >= quality_constraint

        # problem solving
        if self.time_limit is not None:
            ilp_problem.solve(pl.PULP_CBC_CMD(timeLimit=self.time_limit))
        else:
            ilp_problem.solve()

        models_chosen, costs_chosen = self.resolve_ILP(ilp_problem, c, v, models)
        return models_chosen, costs_chosen


    def _assign(self, batch: Dict[str, Any]):
        probs = batch["prob"]
        models = list(probs[0].keys())
        costs = batch["cost"]

        # ensure consistent ordering of models
        if self.model_prob is not None:
            self.model_prob = {key: self.model_prob[key] for key in models}

        costs_chosen = []
        models_chosen = []
        if len(probs) == 1:
            probs = probs[0]
            costs = costs[0]
            cheapest_model = None
            cheapest_cost = np.inf
            for model in models:
                if probs[model] > self.threshold and costs[model] < cheapest_cost:
                    cheapest_model = model
                    cheapest_cost = costs[model]
            if cheapest_cost > self.cost_constraint:
                cheapest_model = None
            if cheapest_model is None:
                cheapest_cost = 0
            costs_chosen.append(cheapest_cost)
            models_chosen.append(cheapest_model)
        else:
            models_chosen, costs_chosen = self.solve_ILP(probs, models, costs, self.model_prob)
        return models_chosen, costs_chosen
