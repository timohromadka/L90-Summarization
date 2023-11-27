from collections import defaultdict, Counter
import logging
import math
import numpy as np
import os
import pickle
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
from torch.utils.data import DataLoader, TensorDataset, dataset
from transformers import BertTokenizer
from tqdm import tqdm

from positional_encoding import PositionalEncoding
from bpe import Encoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AbstractiveSummarizer(nn.Module):
    
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        d_hid: int, 
        nlayers: int, 
        dropout: float = 0.1,
        tokenizer: str = 'wordpiece'):
        
        super(AbstractiveSummarizer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.SOS = '<sos>'
        self.EOS = '<eos>'
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.init_vocabulary(method=tokenizer)
        self.ntoken = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(self.ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.model = nn.Transformer(
            d_model, 
            nhead, 
            nlayers, 
            nlayers, 
            d_hid, 
            dropout)
        self.linear = nn.Linear(d_model, self.ntoken)

        self.init_weights()
        
        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
       
    def init_vocabulary(self, method='wordpiece', vocab_size=4000, train_articles=None, vocab_file='vocab.json'):
        """
        Initialize the vocabulary with the specified tokenization method.
        """
        logger.info(f'Initializing tokenization vocabulary with method: {method}.')
        if method == 'wordpiece':
            # Initialize with a pre-trained model's tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased', 
                do_lower_case=True,
            )

            # # Define custom special tokens
            # self.SOS = '<sos>'
            # self.EOS = '<eos>'
            # special_tokens_dict = {'additional_special_tokens': [self.SOS, self.EOS]}
            # self.tokenizer.add_special_tokens(special_tokens_dict)

            # Resize model embeddings to accommodate new tokens
            # Assuming you have a model variable. Uncomment and modify according to your model.
            # model.resize_token_embeddings(len(self.tokenizer))

            logger.info('WordPiece tokenizer initialized with custom special tokens.')

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.model(src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        output = self.linear(output)
        return output
        
    def train(
        self, 
        train_dataloader,
        val_dataloader,
        learning_rate=0.001, 
        grad_acc=1, 
        num_epochs=10, 
        keep_best_n_models=2
        ):
        """
        X: list of sentences (i.e., articles)
        y: list of sentences (i.e., summaries)
        learning_rate: learning rate for Adam optimizer
        batch_size: batch size for training
        grad_acc: number of gradient accumulation steps to sum over. Set to > 1 to simulate larger batch sizes on memory-limited machines.
        num_epochs: number of epochs to train for
        keep_best_n_models: number of best models to keep (based on validation performance)
        """
        
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        best_model_paths = []
        best_model_scores = []

        logger.info('Beginning training.')
        for epoch in range(num_epochs):
            for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Training epoch {}".format(epoch)):
                input_ids, attention_mask, labels, _ = [x.to(self.device) for x in batch]

                optimizer.zero_grad()
                if input_ids.size(0) != labels.size(0):
                    print(f"Mismatched batch sizes at batch {batch_idx}: input_ids has {input_ids.size(0)} elements, labels has {labels.size(0)} elements")
                    return
                
                outputs = self.forward(input_ids, labels, src_key_padding_mask=(attention_mask==0).T)
                loss = self.compute_loss(outputs, labels) / grad_acc
                loss.backward()

                if (batch_idx + 1) % grad_acc == 0:
                    optimizer.step()

            score = self.compute_validation_score(val_dataloader)

            # Save the model, if performance has improved (keeping n models saved)
            if len(best_model_scores) < keep_best_n_models or score > min(best_model_scores):
                # Save the model:
                best_model_scores.append(score)
                best_model_path = "model-" + str(epoch) + "_score-" + str(score) + ".pt"
                best_model_paths.append(best_model_path)
                torch.save(self.model.state_dict(), best_model_path)

                # Delete the worst model:
                if len(best_model_scores) > keep_best_n_models:
                    worst_model_index = np.argmin(best_model_scores)
                    
                    os.remove(best_model_paths[worst_model_index])
                    del best_model_paths[worst_model_index]
                    del best_model_scores[worst_model_index]

        # Recall the best model:
        best_model_index = np.argmax(best_model_scores)
        self.model.load_state_dict(torch.load(best_model_paths[best_model_index]))

    def preprocess(self, X, y, d_model, batch_size=32):
        logger.info('Tokenizing and converting text to indices using WordPiece in batches.')
        # Tokenize articles and summaries in batches
        tokenized_articles = self.tokenizer(X, add_special_tokens=True, padding='max_length', max_length=d_model, truncation=True, return_tensors='pt')
        tokenized_summaries = self.tokenizer(y, add_special_tokens=True, padding='max_length', max_length=d_model, truncation=True, return_tensors='pt')

        # Create a dataset and DataLoader
        dataset = TensorDataset(tokenized_articles['input_ids'], tokenized_articles['attention_mask'], 
                                tokenized_summaries['input_ids'], tokenized_summaries['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=batch_size)

        return dataloader
        

    def compute_loss(self, outputs, labels):
        """
        outputs: model outputs
        labels: ground truth labels
        """
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs.view(-1, self.ntoken), labels.view(-1))
        return loss


    def decode(self, tokens):
        """
        tokens: list of token indices
        """

        return [self.vocab.itos[token] for token in tokens]

    def compute_validation_score(self, dataloader):
        """
        dataloader: DataLoader object containing validation data
        """

        self.model.eval()

        metric = load_metric('rouge')
        for batch in dataloader:
            input_ids, attention_mask, labels, _ = [x.to(self.device) for x in batch]
            predictions = self.predict(input_ids, attention_mask)
            metric.add_batch(predictions=predictions, references=labels)

        score = metric.compute()
        return score
    
    
    def generate(self, tokenized_X, max_length=50):
        """
        tokenized_X: Tensor of tokenized sentences (i.e., articles)
        max_length: Maximum length of the generated summary
        """
        # Ensure input is on the correct device
        tokenized_X = tokenized_X.to(self.device)

        # Initialize output sequences with <sos> token
        batch_size = tokenized_X.size(0)
        outputs = torch.full((batch_size, 1), self.tokenizer.convert_tokens_to_ids(self.SOS), device=self.device)

        for _ in range(max_length):
            output = self.forward(tokenized_X, outputs)
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            outputs = torch.cat([outputs, next_token], dim=-1)

            # Stop if <eos> token is generated
            if torch.eq(next_token, self.tokenizer.convert_tokens_to_ids(self.EOS)).any():
                break

        return outputs


    def predict(self, input_ids, attention_mask):
        """
        input_ids: Tensor of tokenized sentences (i.e., articles)
        attention_mask: Tensor representing the attention mask for input_ids
        """
        # Ensure the inputs are on the correct device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Generate the summaries
        generated_summaries = self.generate(input_ids)

        # Decode the generated token indices to text
        decoded_summaries = []
        for summary in generated_summaries:
            # Convert tensor to list and remove special tokens
            tokens = summary.tolist()
            tokens = [token for token in tokens if token not in [self.tokenizer.convert_tokens_to_ids(self.SOS), self.tokenizer.convert_tokens_to_ids(self.EOS)]]

            # Decode tokens to string
            decoded_summary = self.tokenizer.decode(tokens, skip_special_tokens=True)
            decoded_summaries.append(decoded_summary)

        return decoded_summaries


            
            
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