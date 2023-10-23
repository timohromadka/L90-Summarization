import tqdm

class ExtractiveSummarizer:

    def train(self, X, y):
        """
        X: list of lists of strings
        y: list of yes/no decision for each string (as boolean)
        """
        
        """
        TODO: Implement me!
        """

        pass

    def predict(self, X):
        """
        X: list of lists of strings
        """
        
        for article in tqdm.tqdm(X, desc="Running extractive summarizer"):
            """
            TODO: Implement me!
            """
            yield "TODO: Implement me!"