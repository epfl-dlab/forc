import torch
import numpy as np


class MetaCollator:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.params = kwargs

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def collate_fn(self, batch):
        collated_batch = {}
        max_length = self.params.get("max_length", None)
        input_field = self.params.get("input", None)
        input_text = [sample[input_field] for sample in batch]
        tokenizer_output = self.tokenizer(
            input_text,
            return_tensors="pt",
            return_attention_mask=True,
            padding=self.params["padding"],
            max_length=max_length,
            truncation=self.params["truncation"],
        )
        for k, v in tokenizer_output.items():
            collated_batch[k] = v
        collated_batch["labels"] = torch.tensor([sample["labels"] for sample in batch]).reshape(-1,self.params["num_outputs"])

        if self.params.get("target_padding_token_id", None) is not None:
            tgt_input_ids = collated_batch["tgt_input_ids"]
            tgt_input_ids.masked_fill_(
                tgt_input_ids == self.tokenizer.pad_token_id, self.params["target_padding_token_id"]
            )
        collated_batch["raw"] = batch

        return collated_batch
