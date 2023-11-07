import logging
import tqdm
from typing import List
import random
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sigmoid(x, clip=False):
    if clip:
        x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def neg_log_likelihood(y_pred: List[float], y_true: List[float], clip=False) -> float:
    # Binary cross-entropy loss
    if clip:
        epsilon = 1e-15  # Small value to prevent divide by zero in log
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class ExtractiveSummarizer:
    
    def __init__(self):
        # Initialize the weights and bias with small random values
        self.weights = None
        self.bias = np.random.random()
        
        self.tfidf_vectorizer = None
        
    def set_weights(self, input_dim):
        self.weights = np.random.randn(input_dim)    
        
    def forward(self, x):
        # Linear combination followed by sigmoid
        return sigmoid((x @ self.weights) + self.bias)

    def compute_loss(self, y_pred, y_true):
        # Binary cross-entropy loss
        return neg_log_likelihood(y_pred, y_true)

    def preprocess(self, X):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        logger.info(f'Preprocessing data!')
        split_articles = [[s.strip() for s in x.split('.')] for x in X]
        logger.info(f'Done!')
        
        return split_articles
    
    
    def featurize(self, X, args):
        logger.info('Transforming data to features!')
        
        X = self.preprocess(X)
        
        if args.method in ['random', 'first', 'first_and_last']:
            return X
        
        elif args.method == 'custom_features':
            all_feature_vectors = []
            for article in tqdm.tqdm(X, desc='Transforming data to features!'):
                sentence_vectors = []
                for i, sentence in enumerate(article):
                    # Compute the additional features
                    custom_features = np.array([
                        len(sentence),
                        len(sentence.split()),
                        i,
                        len(re.findall(r'\d', sentence)),  # Number of digits
                        sentence.count(':'),               # Number of colons
                        len(re.findall(r'[^\w\s]', sentence))  # Number of punctuation characters
                    ])
                    sentence_vectors.append(custom_features)
                all_feature_vectors.append(sentence_vectors)
            logger.info('Done!')
            
            return all_feature_vectors
        
        elif args.method == 'random_features':
            # simply see how well random feature vectors perform
            features = [[np.array([random.random() for _ in range(100)]) for sentence in article] for article in X]
            return features
        
        elif args.method in ['tfidf', 'pairwise_tfidf']:
            # implement TFIDF features
            # 1) fit tfidf vectorizer
            # 2) vectorize features!

            all_sentences = [sentence for article in X for sentence in article]
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000
                )
            self.tfidf_vectorizer.fit(all_sentences)
            logger.info('TF-IDF Vectorizer fitted.')
                
            # if args.method == 'tfidf':
            features = [self.tfidf_vectorizer.transform(article).toarray() for article in tqdm.tqdm(X, desc='Computing TF-IDF features')]

            return features
            
        elif args.method in ['embeddings', 'pairwise_embeddings']:
            model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

            all_sentences = [sentence for article in X for sentence in article]

            batch_size = 2048  # make sure batch can fit into python memory
            features = []

            # Iterate over batches of sentences
            for batch_start in tqdm.tqdm(range(0, len(all_sentences), batch_size), desc='Computing Sentence-BERT embeddings'):
                batch_sentences = all_sentences[batch_start:batch_start + batch_size]
                batch_features = model.encode(
                    batch_sentences,
                    convert_to_tensor=False,
                    show_progress_bar=False
                )
                features.extend(batch_features)
            
            logger.info('Sentence-BERT embeddings computed.')
            
            # reshape features to match the original articles' structure
            article_features_list = []
            idx = 0
            for article in X:
                article_length = len(article)
                article_features = features[idx:idx + article_length]
                article_features_list.append(article_features)
                idx += article_length
            
            return article_features_list
                
        else:
            logger.warning(f'Unrecognized argument: {args.method=}')
        

    def train(self, X_train, y_train, X_articles, eval_summaries, args, epochs=3, gd='sgd', lr=0.01):
        """
        X: list of list of sentences (i.e., comprising an article)
        y: list of yes/no decision for each sentence (as boolean)
        """

        for article, decisions in tqdm.tqdm(zip(X_train, y_train), desc="Validating data shape", total=len(X_train)):
            assert len(article) == len(decisions), "Article and decisions must have the same length"

        logger.info(f'Beginning Training!')
        for epoch in range(epochs):
            combined = list(zip(X_train, y_train))
            random.shuffle(combined)
            X_train[:], y_train[:] = zip(*combined)
            
            total_loss = 0
            count = 0
            
            if gd == 'sgd':
                for i in tqdm.tqdm(range(len(X_train)), desc='Iterating over each article'): # for each article
                    for j in range(len(X_train[i])): # for each sentence in each article
                        y_pred = self.forward(X_train[i][j])
                        error = y_pred - y_train[i][j]

                        # Compute gradient
                        dw = np.dot(error, X_train[i][j].T)
                        db = error

                        # Update weights and bias with SGD
                        self.weights -= lr * dw
                        self.bias -= lr * db
                        
                        # Accumulate loss
                        total_loss += self.compute_loss(y_pred, y_train[i][j])
                        count += 1

            average_loss = total_loss / count if count > 0 else 0
            logger.info(f'Average loss for epoch {epoch}: {average_loss:.6f}!')
            
            # logger.info('Validating on validation set!')
            
                

    def predict(self, article_features, articles, args, classification_threshold=0.10):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        
        articles = self.preprocess(articles) # we must perform this additional step
        
        for article_feature, article in tqdm.tqdm(zip(article_features, articles), desc="Running extractive summarizer"):
            
            if args.method == 'random':
                k = 3
                sentence_scores = [random.random() for _ in article]
            
                top_k_idxs = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:k]
                top_sentences = [article[i] for i in top_k_idxs]
                summary = ' . '.join(top_sentences)
                
                yield sentence_scores, summary
                
            elif args.method == 'first':
                sentence_scores = [1 if i < 3 else 0 for i, _ in enumerate(article)]
                summary = ' . '.join([sentence for i, sentence in enumerate(article) if i < 3])
                
                yield sentence_scores, summary
                
            elif args.method == 'first_and_last':
                sentence_scores = [1 if i < 3 else 0 for i, _ in enumerate(article)]
                summary = ' . '.join([sentence for i, sentence in enumerate(article) if i < 3 or i == len(article)-1])
                
                yield sentence_scores, summary
                
                
            elif args.method == 'pairwise':
                # finish this if time permits
                # Pairwise cosine similarity matching
                # Each sentence gets matched with 
                yield "", ""
                
            else:                  
                # use logistic regression + feature vectors for classification
                sentence_scores = [self.forward(feature_vec) for feature_vec in article_feature]
                final_sentences = [sent for sent, score in zip(article, sentence_scores) if score > classification_threshold] # keep just those that were deemed important
                summary = ". ".join(final_sentences)
                
                # we will also yield sentence scores, in case we would like to test with a new classification threshold
                yield sentence_scores, summary
    
    def predict_only(self, article_scores, articles, args, classification_threshold=0.10):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        
        articles = self.preprocess(articles) # we must perform this additional step
        
        for article_score, article in tqdm.tqdm(zip(article_scores, articles), desc="Running extractive summarizer on pre-computed sentence scores"):
            final_sentences = [sent for sent, score in zip(article, article_score) if score > classification_threshold] # keep just those that were deemed important
            summary = ". ".join(final_sentences)
            
            # we will also yield sentence scores, in case we would like to test with a new classification threshold
            yield article_score, summary    