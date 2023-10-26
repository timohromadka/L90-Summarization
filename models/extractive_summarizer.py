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

    def train(self, x, y, lr=0.01, epochs=1000):
        for epoch in range(epochs):
            # TODO:
            # implement the following:
            # 1) batch gradient descent
            # 2) stochastic gradient descent (start with this)
            # 3) Mini-batch gradient descent
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

    def fit_vectorizer(self, X):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        all_sentences = [sentence for article in X for sentence in article]
        logger.info(f'Now fitting TF-IDF!')
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
    
        all_feature_vectors = []

        for article in X:
            # Use the trained vectorizer to transform the sentences into TF-IDF vectors
            tfidf_matrix = self.vectorizer.transform(article).toarray()
            all_feature_vectors.append(tfidf_matrix)

        return all_feature_vectors

    def preprocess(self, X):
        """
        X: list of list of sentences (i.e., comprising an article)
        TODO: use NLTK sentence splitter instead
        TODO: add sentence tokenization
        TODO: all words to lowercase
        TODO: remove stopwords (optional) - lets see how this performs w/ and w/out
        """
        
        split_articles = [[s.strip() for s in x.split('.')] for x in X]
        
        return split_articles


    def train(self, X, y):
        """
        X: list of list of sentences (i.e., comprising an article)
        y: list of yes/no decision for each sentence (as boolean)
        """
        
        for article, decisions in tqdm.tqdm(zip(X, y), desc="Validating data shape", total=len(X)):
            assert len(article) == len(decisions), "Article and decisions must have the same length"

        """
        TODO: Implement me!
        """
        
            

    def predict(self, X):
        """
        X: list of list of sentences (i.e., comprising an article), each sentence being a feature vector (already preprocessed)
        """
        
        features = self.featurize(X) # featurize articles here in order to preserve original article text
        
        for article, feature_list in tqdm.tqdm(zip(X, features), desc="Running extractive summarizer"):
            """
            TODO: Implement me!
            """

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
            
            
# function logistic_regression(X, y, alpha, num_iterations):
#     m, n = shape(X)   # m is number of samples, n is number of features
#     W = initialize_weights(n)
#     b = 0  # Bias term

#     for i in range(num_iterations):
#         Z = dot_product(X, W) + b
#         A = sigmoid(Z)  # Activation, where sigmoid(z) = 1 / (1 + exp(-z))

#         # Compute the cost using binary cross-entropy loss
#         cost = -1/m * sum(y * log(A) + (1-y) * log(1-A))

#         # Gradient computation
#         dW = 1/m * dot_product(transpose(X), A-y)
#         db = 1/m * sum(A-y)

#         # Update weights using gradient descent
#         W = W - alpha * dW
#         b = b - alpha * db

#         if i % some_interval == 0:  # You can set some_interval to 100 or other value for logging
#             print("Cost after iteration", i, ":", cost)

#     return W, b

# function predict(X, W, b):
#     Z = dot_product(X, W) + b
#     A = sigmoid(Z)
#     predictions = empty_like(A)
    
#     for i in range(length(A)):
#         if A[i] > 0.5:
#             predictions[i] = 1
#         else:
#             predictions[i] = 0

#     return predictions

# function initialize_weights(n):
#     return zeros(n)  # Initialize weights to zeros or small random values

# function sigmoid(z):
#     return 1 / (1 + exp(-z))

# # Using the functions
# X_train, y_train = load_training_data()
# alpha = 0.01
# num_iterations = 1000

# W, b = logistic_regression(X_train, y_train, alpha, num_iterations)

# X_test = load_test_data()
# predictions = predict(X_test, W, b)