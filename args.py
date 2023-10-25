import argparse
from datetime import datetime

parser = argparse.ArgumentParser(
    description="Tyrsak Online Reservation System."
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

