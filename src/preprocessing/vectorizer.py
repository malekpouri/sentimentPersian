# src/preprocessing/vectorizer.py

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import re


class TextVectorizer:
    def __init__(self, max_words=10000, max_len=100, embedding_dim=100):
        """
        Initialize text vectorizer
        Args:
            max_words: Maximum number of words to keep
            max_len: Maximum length of each sequence
            embedding_dim: Dimension of word embeddings
        """
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.word_tokenizer = Tokenizer(num_words=max_words)
        self.char_tokenizer = Tokenizer(char_level=True)
        self.word2vec_model = None

    def fit_word_tokenizer(self, texts):
        """Fit word tokenizer on texts"""
        self.word_tokenizer.fit_on_texts(texts)

    def fit_char_tokenizer(self, texts):
        """Fit character tokenizer on texts"""
        self.char_tokenizer.fit_on_texts(texts)

    def train_word2vec(self, texts, min_count=1, window=5, workers=4):
        """Train Word2Vec model on texts"""
        # Tokenize sentences into words
        sentences = [text.split() for text in texts]

        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            sentences,
            vector_size=self.embedding_dim,
            window=window,
            min_count=min_count,
            workers=workers
        )

    def create_embedding_matrix(self):
        """Create embedding matrix from Word2Vec model"""
        embedding_matrix = np.zeros((self.max_words, self.embedding_dim))
        for word, i in self.word_tokenizer.word_index.items():
            if i < self.max_words:
                try:
                    embedding_vector = self.word2vec_model.wv[word]
                    embedding_matrix[i] = embedding_vector
                except KeyError:
                    continue
        return embedding_matrix

    def texts_to_sequences(self, texts, char_level=False):
        """Convert texts to sequences of indices"""
        tokenizer = self.char_tokenizer if char_level else self.word_tokenizer
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len)
        return padded_sequences

    def prepare_data(self, texts, char_level=False):
        """Prepare data for both word and character level"""
        if char_level:
            return self.texts_to_sequences(texts, char_level=True)
        return self.texts_to_sequences(texts)

    def get_vocab_size(self, char_level=False):
        """Get vocabulary size"""
        tokenizer = self.char_tokenizer if char_level else self.word_tokenizer
        return min(len(tokenizer.word_index) + 1, self.max_words)


# Usage example
if __name__ == "__main__":
    # Example texts
    texts = [
        "این یک متن نمونه است",
        "این متن برای تست است"
    ]

    # Initialize vectorizer
    vectorizer = TextVectorizer(max_words=1000, max_len=50, embedding_dim=100)

    # Fit tokenizers
    vectorizer.fit_word_tokenizer(texts)
    vectorizer.fit_char_tokenizer(texts)

    # Train Word2Vec
    vectorizer.train_word2vec(texts)

    # Get word sequences
    word_sequences = vectorizer.prepare_data(texts)

    # Get character sequences
    char_sequences = vectorizer.prepare_data(texts, char_level=True)

    print("Word sequences shape:", word_sequences.shape)
    print("Char sequences shape:", char_sequences.shape)