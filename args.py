import argparse
from datetime import datetime

parser = argparse.ArgumentParser(
    description="Extractive Summarizer for L90 Assignment #2."
)

parser.add_argument(
    '--train_data', 
    type=str, 
    default='data/train.greedy_sent.json',
)

parser.add_argument(
    '--eval_data', 
    type=str,
    default='data/validation.json'
)

parser.add_argument(
    '--test_data', 
    type=str,
    default='data/test.json'
)

parser.add_argument(
    '--method', 
    type=str,
    default='random',
    choices=['random', 'first', 'first_and_last', 'custom_features', 'tfidf', 'embeddings', 'pairwise', 'random_features']
)

# use the sklearn logistic regression classifier to test how it performs
parser.add_argument(
    '--predict_only', 
    action='store_true',
    default=False
)