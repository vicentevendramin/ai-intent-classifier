from sklearn.feature_extraction.text import CountVectorizer


class BoWVectorizer:
    """
    A class to vectorize text data using the Bag of Words (BoW) technique.
    """
    def __init__(self, **kwargs):
        """
        Initializes the BoWVectorizer with opitional parameters for CountVectorizer.
        """
        self.vectorizer = CountVectorizer(**kwargs)
        self.vocabulary_ = None

    def fit(self, raw_documents):
        """
        Learns the vocabulary dictionary and returns the fitted vectorizer.
        """
        self.vectorizer.fit(raw_documents)
        self.vocabulary_ = self.vectorizer.vocabulary_
        return self
    
    def transform(self, raw_documents):
        """
        Transform documents to document-term matrix.
        """
        return self.vectorizer.transform(raw_documents)
    
    def fit_transform(self, raw_documents):
        """
        Learns the vocabulary dictionary and returns the document-term matrix.
        This is equivalent to fit followed by transform, but more efficiently implemented.
        """
        self.fit(raw_documents)
        return self.transform(raw_documents)
    
    def get_feature_names(self):
        """
        Returns the feature names (words in the vocabulary).
        """
        return self.vectorizer.get_feature_names_out()


if __name__ == '__main__':
    corpus = [
        "I want to buy this house.",
        "What is the price of that apartment?",
        "Are there any available properties?",
    ]

    # Initialize the BoWVectorizer
    bow_vectorizer = BoWVectorizer()

    # Fit and transform the corpus
    bow_matrix = bow_vectorizer.fit_transform(corpus)

    # Get the vocabulary
    vocabulary = bow_vectorizer.vocabulary_
    print("Vocabulary:", vocabulary)

    # Get the feature names (words)
    feature_names = bow_vectorizer.get_feature_names()
    print("Feature Names:", feature_names)

    # Print the BoW matrix (sparce representation)
    print("BoW Matrix (sparce):\n", bow_matrix)

    # Print the BoW matrix (dense representation)
    print("BoW Matrix (dense):\n", bow_matrix.toarray())
