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
    choices=[
        'random', # randomly assign scores to sentences
        'random_features', # randomly assign features to sentences
        'first',
        'first_and_last', 
        
        'custom_features', 
        'tfidf', 
        'embeddings', 
        'pairwise_tfidf',
        'pairwise_embeddings'
    ]
)

# use the sklearn logistic regression classifier to test how it performs
parser.add_argument(
    '--predict_only', 
    action='store_true',
    default=False
)

parser.add_argument(
    '--classification_threshold', 
    type=float,
    default=0.1
)

parser.add_argument(
    '--epochs', 
    type=int,
    default=5
)

parser.add_argument(
    '--lr', 
    type=float,
    default=5
)

parser.add_argument(
    '--gd', 
    type=str,
    default='sgd',
    choices=['sgd', 'batch']
)