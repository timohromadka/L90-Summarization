import argparse
import json

from torchtext.datasets import WikiText2

from models.abstractive_summarizer import AbstractiveSummarizer

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--train_data', type=str, default='data/train.json')
    args.add_argument('--validation_data', type=str, default='data/validation.json')
    args.add_argument('--test_data', type=str, default='data/test.json')
    args = args.parse_args()
    
    hparams = {
        # "train_data": "data/train.json",
        # "validation_data": "data/validation.json",
        # "eval_data": "data/test.json",
        'd_model': 512,
        'nhead': 4,
        'd_hid': 512,
        'nlayers': 2,
        'dropout': 0.1,
        'tokenizer': 'wordpiece',
        'learning_rate': 0.001,
        'num_epochs': 10,
        'grad_acc': 1,
        'batch_size': 32
    }

    model = AbstractiveSummarizer(
        hparams['d_model'], 
        hparams['nhead'], 
        hparams['d_hid'], 
        hparams['nlayers'], 
        hparams['dropout'],
        hparams['tokenizer'],
        )

    with open(args.train_data, 'r') as f:
        train_data = json.load(f)

    with open(args.validation_data, 'r') as f:
        validation_data = json.load(f)

    train_articles = [article['article'] for article in train_data]
    train_summaries = [article['summary'] for article in train_data]

    val_articles = [article['article'] for article in validation_data]
    val_summaries = [article['summary'] for article in validation_data]

    # model.init_vocabulary(train_articles)
    model.train(
        train_articles, 
        train_summaries, 
        val_articles,
        val_summaries,
        learning_rate=hparams['learning_rate'],
        batch_size=hparams['batch_size'], 
        grad_acc=hparams['grad_acc'], 
        num_epochs=hparams['num_epochs'],
    )

    with open(args.test_data, 'r') as f:
        test_data = json.load(f)

    test_articles = [article['article'] for article in test_data]
    summaries = model.predict(test_articles)
    test_out_data = [{'article': article, 'summary': summary} for article, summary in zip(test_articles, summaries)]

    print(json.dumps(test_out_data, indent=4))

if __name__ == '__main__':
    main()