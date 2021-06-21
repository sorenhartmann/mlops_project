from typing import List

import torch
from transformers import ConvBertTokenizer


class Tokenizer:
    def __init__(self):

        self.tokenizer = ConvBertTokenizer.from_pretrained(
            "YituTech/conv-bert-base"
        )

    def tokenize(self, strings: List[str]):

        tokenized_output = self.tokenizer(strings)

        input_ids = torch.tensor(tokenized_output.input_ids)
        attention_mask = torch.tensor(tokenized_output.attention_mask)
        token_type_ids = torch.tensor(tokenized_output.token_type_ids)

        return input_ids, attention_mask, token_type_ids
