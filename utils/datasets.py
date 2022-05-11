import tqdm
import random
import numpy as np

import torch


class MLMDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_path, tokenizer, encoding="utf-8"):
        with open(corpus_path, "r", encoding=encoding) as f:
            self.txt_in = [f_i for f_i in f.read().split('\n') if f_i]
        self.txt_len = len(self.txt_in)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.txt_in)

    def __getitem__(self, idx):
        y_true = self.tokenizer(self.txt_in[idx], padding='max_length', return_tensors="pt")
        y_mask = self.mlm(y_true)

        return y_mask, y_true

    def mlm(self, y_true):
        y_true_input_ids = y_true['input_ids']
        total_length = int(torch.sum(y_true_input_ids != 0))
        y_mask = y_true_input_ids.squeeze().clone().detach()
        for i in range(1, total_length - 1):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                # 80% masking
                if prob < 0.8:
                    y_mask[i] = self.tokenizer.mask_token_id
                # 10% random token
                elif prob < 0.9:
                    y_mask[i] = random.randrange(len(self.tokenizer))
                # 10% the current token
                else:
                    y_mask[i] = y_true_input_ids[0, i]

        return y_mask.unsqueeze(0)
