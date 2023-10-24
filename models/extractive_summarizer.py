import tqdm
import random

class ExtractiveSummarizer:

    def preprocess(self, X):
        """
        X: list of list of sentences (i.e., comprising an article)
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

    def predict(self, X, k=3):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        
        for article in tqdm.tqdm(X, desc="Running extractive summarizer"):
            """
            TODO: Implement me!
            """

            # Randomly assign a score to each sentence. 
            # This is just a placeholder for your actual model.
            sentence_scores = [random.random() for _ in article]

            # Pick the top k sentences as summary.
            # Note that this is just one option for choosing sentences.
            top_k_idxs = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:k]
            top_sentences = [article[i] for i in top_k_idxs]
            summary = ' . '.join(top_sentences)
            
            yield summary