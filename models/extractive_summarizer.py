import tqdm
import logging
import random
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def neg_log_likelihood(y: float, y_pred: float) -> float:
    return -((y * np.log(y_pred)) + ((1 - y) * np.log(1 - y_pred)))

def neg_log_likelihood(y_pred: List[float], y_true: List[float]) -> float:
    # Binary cross-entropy loss
    # TODO: should -npmean be out in front?
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class ExtractiveSummarizer:
    
    def __init__(self, input_dim=None):
        # Initialize the weights and bias with small random values
        self.weights = input_dim
        self.bias = 0.0
        self.vectorizer = TfidfVectorizer(max_features=100)
    
    def set_weights(self, input_dim):
        self.weights = np.random.randn(input_dim)

    def forward(self, x):
        # Linear combination followed by sigmoid
        return sigmoid((x @ self.weights) + self.bias)

    def compute_loss(self, y_pred, y_true):
        # Binary cross-entropy loss
        return neg_log_likelihood(y_pred, y_true)

    def train(self, x, y, lr=0.01, epochs=1000, method='sgd'):
        for epoch in range(epochs):
            # TODO:
            # implement the following:
            # 1) batch gradient descent
            # 2) stochastic gradient descent (start with this)
            # 3) Mini-batch gradient descent
            if method == 'sgd':
                for i in range(len(x)):
                    y_pred = self.forward(x[i])
                    error = y_pred - y[i]

                    # Compute gradient
                    dw = error * x[i]
                    db = error

                    # Update weights and bias with SGD
                    self.weights -= lr * dw
                    self.bias -= lr * db

                # Print loss every 100 epochs
                if epoch % 100 == 0:
                    y_preds = self.forward(x)
                    loss = self.compute_loss(y_preds, y)
                    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')
                    
            elif method == 'batch':
                pass
            elif method == 'mini_batch':
                pass

    def fit_vectorizer(self, X):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        logger.info(f'Now fitting TF-IDF!')
        all_sentences = [sentence for article in X for sentence in article]
        self.vectorizer.fit(all_sentences)
        logger.info('Done!')


    def featurize(self, X):
        """
        X: list of list of sentences (i.e., comprising an article)
        return: list of list of features
        
        TODO: IDEAS FOR FEATURES (in .txt)
        - tfidf
        
        TODO: add scaling if necessary
        """
        logger.info(f'Transforming data to features!')
        
        all_feature_vectors = []

        for article in X:
            # Use the trained vectorizer to transform the sentences into TF-IDF vectors
            tfidf_matrix = self.vectorizer.transform(article).toarray()
            all_feature_vectors.append(tfidf_matrix)
            
        logger.info(f'Done!')

        return all_feature_vectors

    def preprocess(self, X):
        """
        X: list of list of sentences (i.e., comprising an article)
        TODO: use NLTK sentence splitter instead
        TODO: add sentence tokenization
        TODO: all words to lowercase
        TODO: remove stopwords (optional) - lets see how this performs w/ and w/out
        """
        logger.info(f'Preprocessing data!')
        split_articles = [[s.strip() for s in x.split('.')] for x in X]
        logger.info(f'Done!')
        return split_articles


    # def train(self, X, y):
    #     """
    #     X: list of list of sentences (i.e., comprising an article)
    #     y: list of yes/no decision for each sentence (as boolean)
    #     """
        
    #     for article, decisions in tqdm.tqdm(zip(X, y), desc="Validating data shape", total=len(X)):
    #         assert len(article) == len(decisions), "Article and decisions must have the same length"

    #     """
    #     TODO: Implement me!
    #     """
        
            

    def predict(self, X):
        """
        X: list of list of sentences (i.e., comprising an article), each sentence being a feature vector (already preprocessed)
        """
        
        features = self.featurize(X) # featurize articles here in order to preserve original article text
        
        for article, feature_list in tqdm.tqdm(zip(X, features), desc="Running extractive summarizer"):

            # =================================================
            # NAIVE METHOD
            
            # Randomly assign a score to each sentence. 
            # This is just a placeholder for your actual model.
            # sentence_scores = [random.random() for _ in article]

            # # Pick the top k sentences as summary.
            # # Note that this is just one option for choosing sentences.
            # k = 3
            # top_k_idxs = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:k]
            # top_sentences = [article[i] for i in top_k_idxs]
            
            # summary = ' . '.join(top_sentences)
            
            # =================================================
            # SUPERVISED METHOD
            
            sentence_scores = [self.forward(features) for features in feature_list]
            final_sentences = [sent for sent, score in zip(article, sentence_scores) if score > 0.5] # keep just those that were deemed important
            summary = ". ".join(final_sentences)
            
            yield summary