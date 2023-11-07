import argparse
import json
import logging
import os
import pickle
import tqdm

from args import parser
from models.extractive_summarizer import ExtractiveSummarizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess(X):
    """
    X: list of list of sentences (i.e., comprising an article)
    """
    
    split_articles = [[s.strip() for s in x.split('.')] for x in X]
    return split_articles
       
def load_or_featurize(data_type, articles, model, args):
    pkl_path = f'pickled_data/{args.method}_{data_type}.pkl'
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as file:
            featurized_articles = pickle.load(file)
    else:
        featurized_articles = model.featurize(articles, args)
        with open(pkl_path, 'wb') as file:
            pickle.dump(featurized_articles, file, pickle.HIGHEST_PROTOCOL)
    return featurized_articles

def main():
    args = args = parser.parse_args()
    model = ExtractiveSummarizer()
       
        
    # ==========================================
    # DATA SETUP
    # ==========================================
    with open(args.train_data, 'r') as f:
        train_data = json.load(f)

    with open(args.eval_data, 'r') as f:
        eval_data = json.load(f)
        
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
        
        
        
    train_articles = [article['article'] for article in train_data]
    train_highlight_decisions = [article['greedy_n_best_indices'] for article in train_data]
    
    eval_articles = [article['article'] for article in eval_data]
    eval_summaries = [article['summary'] for article in eval_data]
    
    test_articles = [article['article'] for article in test_data]
    
    # load the features if they're available
    # if not, generate the features and save
    
    featurized_train_articles = load_or_featurize('train', train_articles, model, args)
    featurized_eval_articles = load_or_featurize('eval', eval_articles, model, args)
    featurized_test_articles = load_or_featurize('test', test_articles, model, args)

    def predict_and_save(featurized_articles, articles, args, type):
        # TODO
        # do predict only for DEV SET AS WELL!!!
        # Run predictions using the model
        scores, summaries = zip(*model.predict(featurized_articles, articles, args))

        # Prepare the output data
        eval_out_data = [
            {'article': article, 'summary': summary} 
            for article, summary 
            in zip(articles, summaries)
        ]


        with open(f'{type}_prediction_file_{args.method}.json', 'w') as json_file:
            json.dump(eval_out_data, json_file, indent=4)

        # Save the scores to a pickle file
        with open(f'pickled_scores/{type}_{args.method}_scores.pkl', 'wb') as f:
            pickle.dump(scores, f)

    if args.predict_only:
        score_path = f'pickled_scores/{args.method}_scores.pkl'
        try:
            with open(score_path, 'rb') as file:
                article_scores = pickle.load(file)
        except FileNotFoundError:
            print(f"The file {score_path} was not found with method {args.method}.")
            return
        
        
        # VALIDATION SET
        _, summaries = zip(*model.predict_only(
            article_scores,
            featurized_eval_articles,
            eval_articles,
            args,
            ))
        
        eval_out_data = [
            {'article': article, 'summary': summary} 
            for article, summary 
            in zip(eval_articles, summaries)
            ]
        
        with open(f'eval_prediction_file_{args.method}.json', 'w') as json_file:
            json.dump(eval_out_data, json_file, indent=4)
            
        # TEST SET
        _, summaries = zip(*model.predict_only(
            article_scores,
            featurized_test_articles,
            test_articles,
            args,
            ))
        
        test_out_data = [
            {'article': article, 'summary': summary} 
            for article, summary 
            in zip(test_articles, summaries)
            ]
        
        with open(f'test_prediction_file_{args.method}.json', 'w') as json_file:
            json.dump(test_out_data, json_file, indent=4)

        return
    
    
    
    if args.method not in ['first', 'first_and_last', 'random']:
        model.set_weights(len(featurized_eval_articles[0][0]))
        
        model.train(
            featurized_train_articles, 
            train_highlight_decisions,
            eval_articles,
            eval_summaries,
            args,
            )

            
            
            
    # scores, summaries = zip(*model.predict(
    #     featurized_test_articles,
    #     test_articles,
    #     args
    #     ))
    
    # eval_out_data = [
    #     {'article': article, 'summary': summary} 
    #     for article, summary 
    #     in zip(test_articles, summaries)
    #     ]


    # print(json.dumps(eval_out_data, indent=4))
    
    # with open(f'pickled_scores/{args.method}_scores.pkl', 'wb') as f:
    #     pickle.dump(scores, f)
        
    predict_and_save(featurized_test_articles, test_articles, args, 'test')
    predict_and_save(featurized_eval_articles, eval_articles, args, 'eval')

if __name__ == "__main__":
    main()
    
# EXAMPLE USAGE
# python run_extractive_summarizer.py --train_data data/train.greedy_sent.json --test_data data/test.json --eval_data data/validation.json --method first > test_prediction_file_first_3.json
# python eval.py --eval_data data/test.json --pred_data test_prediction_file_first_3.json