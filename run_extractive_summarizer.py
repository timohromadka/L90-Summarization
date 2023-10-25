import argparse
import json
import tqdm
from models.extractive_summarizer import ExtractiveSummarizer

from args import parser

def main():
    args = parser.parse_args()

    model = ExtractiveSummarizer()

    with open(args.train_data, 'r') as f:
        train_data = json.load(f)

    train_articles = [article['article'] for article in train_data]
    train_highlight_decisions = [article['greedy_n_best_indices'] for article in train_data]

    preprocessed_train_articles = model.preprocess(train_articles)
    model.train(preprocessed_train_articles, train_highlight_decisions)

    with open(args.eval_data, 'r') as f:
        eval_data = json.load(f)

    eval_articles = [article['article'] for article in eval_data]
    preprocessed_eval_articles = model.preprocess(eval_articles)
    summaries = model.predict(preprocessed_eval_articles)
    eval_out_data = [{'article': article, 'summary': summary} for article, summary in zip(eval_articles, summaries)]

    print(json.dumps(eval_out_data, indent=4))

if __name__ == "__main__":
    main()
