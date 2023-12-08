import argparse
import json
import logging
import os
import tqdm

import pandas as pd

from evaluation.rouge_evaluator import RougeEvaluator
import validation.utils as utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@utils.time_func
def main():
    args = argparse.ArgumentParser()
    args.add_argument('--pred_data', type=str, default='eval_prediction_file_custom_features.json')
    args.add_argument('--eval_data', type=str, default='data/validation.json')
    args.add_argument('--run_name', type=str, required=True, help="Name of the current run for tracking")
    args = args.parse_args()

    evaluator = RougeEvaluator()

    with open(args.eval_data, 'r') as f:
        eval_data = json.load(f)

    with open(args.pred_data, 'r') as f:
        pred_data = json.load(f)

    assert len(eval_data) == len(pred_data)

    pred_sums = []
    eval_sums = []
    for eval, pred in tqdm.tqdm(zip(eval_data, pred_data), total=len(eval_data)):
        pred_sums.append(pred['summary'])
        eval_sums.append(eval['summary'])

    scores = evaluator.batch_score(pred_sums, eval_sums)
    
    flattened_scores = {f"{main_key}-{sub_key}": sub_value
                        for main_key, sub_dict in scores.items()
                        for sub_key, sub_value in sub_dict.items()}

    df_scores = pd.DataFrame(list(flattened_scores.values()), index=flattened_scores.keys()).transpose()
    df_scores['run_name'] = args.run_name
    
    csv_file_path = 'results.csv'
    
    if os.path.exists(csv_file_path):
        df_scores.to_csv(csv_file_path, mode='a', header=False, index=False)
    else:
        df_scores.to_csv(csv_file_path, mode='w', header=True, index=False)
    
if __name__ == "__main__":
    main()