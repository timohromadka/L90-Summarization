import argparse
import json
import tqdm
from models.extractive_summarizer import ExtractiveSummarizer

from args import parser

def main():
    # ========================================
    # SETUP
    # ========================================
    args = parser.parse_args()


    with open(args.train_data, 'r') as f:
        train_data = json.load(f)

    train_articles = [article['article'] for article in train_data]
    train_highlight_decisions = [article['greedy_n_best_indices'] for article in train_data]
    
    model = ExtractiveSummarizer()
    preprocessed_train_articles = model.preprocess(train_articles)
    model.fit_vectorizer(preprocessed_train_articles)
    featurized_train_articles = model.featurize(preprocessed_train_articles)
    num_features = len(featurized_train_articles[0][0]) # glance at the first data point's feature vector to get number of features
    model.set_weights(num_features)
    
    # ========================================
    # TRAIN
    # ========================================
    
    model.train(featurized_train_articles, train_highlight_decisions)


    # ========================================
    # EVALUATE
    # ========================================
    with open(args.eval_data, 'r') as f:
        eval_data = json.load(f)

    eval_articles = [article['article'] for article in eval_data]
    preprocessed_eval_articles = model.preprocess(eval_articles)
    # featurized_eval_articles = model.featurize(preprocessed_eval_articles)
    summaries = model.predict(preprocessed_eval_articles)
    eval_out_data = [{'article': article, 'summary': summary} for article, summary in zip(eval_articles, summaries)]

    print(json.dumps(eval_out_data, indent=4))
    
    # ========================================
    # SCRIPTS
    # python run_extractive_summarizer.py --train_data data/train.greedy_sent.json --eval_data data/test.json > test_prediction_file.json
    #
    # python eval.py --eval_data data/test.json --pred_data test_prediction_file.json
    # ========================================

if __name__ == "__main__":
    main()
