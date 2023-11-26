from collections import defaultdict, Counter
import logging
import math
import numpy as np
import os
import re
import json
from tempfile import TemporaryDirectory
from typing import Tuple

from datasets import load_metric
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import dataset
from tqdm import tqdm

from positional_encoding import PositionalEncoding
from bpe import Encoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AbstractiveSummarizer:
    
    def __init__(
        self, 
        ntoken: int, 
        d_model: int, 
        nhead: int, 
        d_hid: int, 
        nlayers: int, 
        dropout: float = 0.1):
        
        super(AbstractiveSummarizer, self).__init__()
        self.SOS = '<sos>'
        self.EOS = '<eos>'
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.embedding = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.model = nn.Transformer(
            d_model, 
            nhead, 
            nlayers, 
            nlayers, 
            d_hid, 
            dropout)
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()
        
        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
       
       
    def init_vocabulary(self, train_articles, method='bpe', vocab_size=4000, vocab_file='vocab.json'):
        """
        X: list of articles (i.e., list of list of sentences)
        vocab_size: desired size of the vocabulary
        """
        if method == 'bpe':
            logger.info('Tokenizing Dataset with BPE Encoding.')
            encoder = Encoder(
                vocab_size=vocab_size, 
                #pct_bpe=0.88,
                silent=False
            )
            text = [word for article in train_articles for word in article.split()]
            encoder.fit(text)
            self.tokenizer = encoder
            logger.info('Done!')

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.model(src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        output = self.linear(output)
        return output
        
    def train(self, X, y, val_X, val_y, learning_rate=0.001, batch_size=32, grad_acc=1, num_epochs=10, keep_best_n_models=2):
        """
        X: list of sentences (i.e., articles)
        y: list of sentences (i.e., summaries)
        learning_rate: learning rate for Adam optimizer
        batch_size: batch size for training
        grad_acc: number of gradient accumulation steps to sum over. Set to > 1 to simulate larger batch sizes on memory-limited machines.
        num_epochs: number of epochs to train for
        keep_best_n_models: number of best models to keep (based on validation performance)
        """

        assert len(X) == len(y), "X and y must have the same length"

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        dataset = self.preprocess(X, y)

        best_model_paths = []
        best_model_scores = []

        for epoch in range(num_epochs):
            # Shuffle the dataset:
            np.random.shuffle(dataset)

            # Split into batches:
            batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]

            # Train on each batch:
            for idx, batch in tqdm.tqdm(enumerate(batches), desc="Training epoch {}".format(epoch)):
                # Compute the loss:
                loss = self.compute_loss(batch) / grad_acc

                # Backprop:
                loss.backward()

                # Handle gradient accumulation:
                if (idx % grad_acc) == 0:
                    optimizer.step()
                    optimizer.zero_grad()


            # Evaluate the model:
            score = self.compute_validation_score(val_X, val_y)

            # Save the model, if performance has improved (keeping n models saved)
            if len(best_model_scores) < keep_best_n_models or score > min(best_model_scores):
                # Save the model:
                best_model_scores.append(score)
                best_model_paths.append("model-" + str(epoch) + "_score-" + str(score) + ".pt")
                torch.save(self.model.state_dict(), best_model_paths[-1])

                # Delete the worst model:
                if len(best_model_scores) > keep_best_n_models:
                    worst_model_index = np.argmin(best_model_scores)
                    
                    os.remove(best_model_paths[worst_model_index])
                    del best_model_paths[worst_model_index]
                    del best_model_scores[worst_model_index]

        # Recall the best model:
        best_model_index = np.argmax(best_model_scores)
        self.model.load_state_dict(torch.load(best_model_paths[best_model_index]))

    def tokenize(self, text):
        """
        Tokenizes the input text.
        text: a string representing a sentence
        Returns a list of tokens
        """
        tokenizer = get_tokenizer('basic_english')
        return [self.vocab[token] for token in tokenizer(text)]


    def preprocess(self, X, y):
        """
        Preprocesses the input data by tokenizing, converting to indices, padding, and adding special tokens.
        X: list of sentences (articles)
        y: list of sentences (summaries)
        Returns a list of tuples, each containing a pair of torch tensors (article, summary)
        """
        # Tokenize the input sentences and add special tokens
        tokenized_X = [[self.SOS] + self.tokenizer.tokenize(article) + [self.EOS] for article in X]
        tokenized_y = [[self.SOS] + self.tokenizer.tokenize(summary) + [self.EOS] for summary in y]

        # Convert tokens to indices
        indexed_X = [self.tokenizer.transform(article) for article in tokenized_X]
        indexed_y = [self.tokenizer.transform(summary) for summary in tokenized_y]

        # Pad and truncate the sequences to a fixed length
        # Accounting for the <sos> and <eos> tokens
        max_length = self.d_model - 2
        padded_indexed_X = [article[:max_length] + [self.EOS] + [self.tokenizer.PAD]*(max_length - len(article)) if len(article) < max_length else article[:max_length] + [self.EOS] for article in indexed_X]
        padded_indexed_y = [summary[:max_length] + [self.EOS] + [self.tokenizer.PAD]*(max_length - len(summary)) if len(summary) < max_length else summary[:max_length] + [self.EOS] for summary in indexed_y]

        # Combine the articles and summaries into pairs
        dataset = list(zip(padded_indexed_X, padded_indexed_y))

        return dataset

        

    def compute_loss(self, batch):
        """
        batch: tensor of token indices. Dimensionality is (batch_size, sequence_length)
        """
        batch_X, batch_y = zip(*batch)
        batch_X = torch.stack(batch_X)
        batch_y = torch.stack(batch_y)

        outputs = self.model(batch_X)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), batch_y.view(-1))

        return loss


    def generate(self, X):
        """
        X: list of sentences (i.e., articles)
        """
        tokenized_X = [torch.tensor([self.vocab.stoi[token] for token in article.split()[:self.max_len]]) for article in X]
        padded_X = pad_sequence(tokenized_X, batch_first=True)

        outputs = self.model(padded_X)
        return torch.argmax(outputs, dim=-1)


    def decode(self, tokens):
        """
        tokens: list of token indices
        """

        return [self.vocab.itos[token] for token in tokens]

    def compute_validation_score(self, X, y):
        """
        X: list of sentences (i.e., articles)
        y: list of sentences (i.e., summaries)
        """

        self.model.eval()

        # compute either loss or ROUGE
        metric = load_metric('rouge')
        predictions = self.predict(X)
        metric.add_batch(predictions=predictions, references=y)
        score = metric.compute()
        return score

    def predict(self, X, k=3):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        
        for article in tqdm.tqdm(X, desc="Running abstractive summarizer"):
            """
            TODO: Implement me!
            """

            output_token_indices = self.generate(article)
            output_tokens = self.decode(output_token_indices)
            summary = ' . '.join(output_tokens)
            
            yield summary
            
            
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)