import numpy as np
import random
from tqdm import tqdm
import json
import math
import time

import spacy
from torch.nn import Transformer
import torch
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer
from torch import Tensor
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k, WikiText2
from typing import Iterable, List, Callable, Iterable
from torch.utils.data import DataLoader, TensorDataset, dataset
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training
from torch.profiler import profile, record_function, ProfilerActivity

from torch.nn.utils.rnn import pad_sequence


from torch.nn.utils.rnn import pad_sequence

class CustomTokenizer:
    def __init__(self, tokenizer_type: str, combined_data: Iterable[str] = None, 
                 path: str = None, vocab_size: int = 30_000, min_frequency: int = 2, emb_size: int = 512):
        self.tokenizer_type = tokenizer_type
        self.emb_size = emb_size  # This should be set to match your model's embedding size
        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

        if tokenizer_type == 'bpe':
            if path is not None:
                self.tokenizer = self.load_bpe_tokenizer(path)
            else:
                self.vocab_size = vocab_size
                self.tokenizer = self.train_bpe_tokenizer(combined_data, vocab_size, min_frequency)
                
        elif tokenizer_type == 'nltk':
            self.token_transform = get_tokenizer('spacy', language='en_core_web_sm')
            self.vocab_transform = self.build_vocab(combined_data)
            self.text_transform = self.sequential_transforms(self.token_transform,
                                                             self.vocab_transform,
                                                             self.tensor_transform)
            self.vocab_transform.set_default_index(self.UNK_IDX)
            self.vocab_size = len(self.vocab_transform)

        else:
            raise ValueError("Unsupported tokenizer type. Choose 'bpe' or 'nltk'.")

    def load_bpe_tokenizer(self, path: str):
        tokenizer = ByteLevelBPETokenizer(
            f"{path}/bpe_tokenizer-vocab.json",
            f"{path}/bpe_tokenizer-merges.txt"
        )
        return tokenizer

    def train_bpe_tokenizer(self, data: Iterable[str], vocab_size: int, min_frequency: int):
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(data, vocab_size=vocab_size, min_frequency=min_frequency)
        tokenizer.save_model(".", "bpe_tokenizer")
        return tokenizer

    def build_vocab(self, data_iter: Iterable) -> None:
        return build_vocab_from_iterator(self.yield_tokens(data_iter),
                                         min_freq=1,
                                         specials=self.special_symbols,
                                         special_first=True)

    def yield_tokens(self, data_iter: Iterable) -> List[str]:
        for data_sample in tqdm(data_iter, desc = 'Tokenizing.'):
            yield self.token_transform(data_sample)

    def sequential_transforms(self, *transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    def tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([self.BOS_IDX]),
                          torch.tensor(token_ids),
                          torch.tensor([self.EOS_IDX])))

    def text_to_tensor(self, sample: str):
        if self.tokenizer_type == 'bpe':
            token_ids = self.tokenizer.encode(sample).ids
            return self.tensor_transform(token_ids)
        elif self.tokenizer_type == 'nltk':
            return self.text_transform(sample)

    def tensor_to_text(self, tensor):
        if self.tokenizer_type == 'bpe':
            token_ids = tensor.tolist()
            if self.BOS_IDX in token_ids:
                token_ids.remove(self.BOS_IDX)
            if self.EOS_IDX in token_ids:
                token_ids.remove(self.EOS_IDX)
            return self.tokenizer.decode(token_ids)
        elif self.tokenizer_type == 'nltk':
            return " ".join([self.vocab_transform.get_itos()[idx] for idx in tensor if idx not in [self.BOS_IDX, self.EOS_IDX, self.PAD_IDX]])

    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_processed = self.text_to_tensor(src_sample)
            tgt_processed = self.text_to_tensor(tgt_sample)

            # Truncate if longer than EMB_SIZE
            src_processed = src_processed[:self.emb_size]
            tgt_processed = tgt_processed[:self.emb_size]

            src_batch.append(src_processed)
            tgt_batch.append(tgt_processed)

        # Pad sequences to be exactly EMB_SIZE
        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.PAD_IDX)

        # Ensure the batch is first and truncate again in case padding made any sequence longer than EMB_SIZE
        src_batch = src_batch.T[:self.emb_size].T
        tgt_batch = tgt_batch.T[:self.emb_size].T

        return src_batch, tgt_batch
