from src.datamodules.abstract import AbstractDataset, AbstractPLDataModule
import os
import pandas as pd
import json


class BinaryMetricDataset(AbstractDataset):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.tokenizer = kwargs.get('tokenizer')
        self.dataset_list_path = kwargs.get("dataset_list_path")
        assert(self.dataset_list_path is not None)
        self._load_data(load_dataset_params=kwargs["load_dataset_params"])

    def _load_data(self, load_dataset_params=None):
        path = os.path.join(load_dataset_params["data_dir"], f"df_{load_dataset_params['split']}.csv")
        df = pd.read_csv(path)
        with open(self.dataset_list_path, 'r') as f:
            dataset_list = json.load(f)
        df = df[df['dataset'].isin(dataset_list)]
        self.queries = df['query'].tolist()
        self.prompts = df['prompt'].tolist()
        self.completions = df['completion'].tolist()
        self.labels = df['label'].tolist()
        self.models = df['model'].tolist()
        self.datasets = df['dataset'].tolist()
        self.query_costs = df['query_cost'].tolist()
        self.prompt_costs = df['prompt_cost'].tolist()
        self.completion_costs = df['completion_cost'].tolist()
        self.total_costs = df['total_cost'].tolist()
        self.ids = df.index.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {"labels": self.labels[idx], "queries":self.queries[idx]}
        if self.models is not None:
            item["models"] = self.models[idx]
        if self.datasets is not None:
            item["datasets"] = self.datasets[idx]
        if self.prompts is not None:
            item["prompts"] = self.prompts[idx]
        if self.completions is not None:
            item["completions"] = self.completions[idx]
        if self.total_costs is not None:
            item["total_costs"] = self.total_costs[idx]
        if self.query_costs is not None:
            item["query_costs"] = self.query_costs[idx]
        if self.prompt_costs is not None:
            item["query_costs"] = self.query_costs[idx]
        if self.completion_costs is not None:
            item["completion_costs"] = self.completion_costs[idx]
        if self.ids is not None:
            item["ids"] = self.ids[idx]
        return item


class BinaryMetricDataModule(AbstractPLDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)