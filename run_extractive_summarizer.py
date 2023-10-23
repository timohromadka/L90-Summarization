import argparse
import json
import tqdm
from models.extractive_summarizer import ExtractiveSummarizer

args = argparse.ArgumentParser()
args.add_argument('--train_data', type=str, default='data/train.greedy_sent.json')
args.add_argument('--eval_data', type=str, default='data/validation.json')
args = args.parse_args()

model = ExtractiveSummarizer()

with open(args.train_data, 'r') as f:
    train_data = json.load(f)

train_articles = [article['article'] for article in train_data]
train_highligt_decisions = [article['greedy_n_best_indices'] for article in train_data]

model.train(train_articles, train_highligt_decisions)

with open(args.eval_data, 'r') as f:
    eval_data = json.load(f)


eval_articles = [article['article'] for article in eval_data]
summaries = model.predict(eval_articles)
eval_out_data = [{'article': article, 'summary': summary} for article, summary in zip(eval_articles, summaries)]

print(json.dumps(eval_out_data, indent=4))