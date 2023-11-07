import argparse
import json
import logging
import tqdm

from evaluation.rouge_evaluator import RougeEvaluator
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@utils.time_func
def main():
    args = argparse.ArgumentParser()
    args.add_argument('--pred_data', type=str, default='data/validation.json')
    args.add_argument('--eval_data', type=str, default='data/validation.json')
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

    for k, v in scores.items():
        print(k)
        print("\tPrecision:\t", v["p"])
        print("\tRecall:\t\t", v["r"])
        print("\tF1:\t\t", v["f"])
    
if __name__ == "__main__":
    main()
