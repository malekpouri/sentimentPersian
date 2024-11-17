# src/models/sentiment_models.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout,
    Bidirectional, Concatenate, GlobalMaxPooling1D
)


class SentimentModels:
    @staticmethod
    def create_lstm_model(
            vocab_size,
            embedding_dim=100,
            max_len=100,
            embedding_matrix=None
    ):
        """Create LSTM model"""
        # Input layer
        inputs = Input(shape=(max_len,))

        # Embedding layer
        if embedding_matrix is not None:
            x = Embedding(
                vocab_size,
                embedding_dim,
                weights=[embedding_matrix],
                trainable=False
            )(inputs)
        else:
            x = Embedding(vocab_size, embedding_dim)(inputs)

        # LSTM layer
        x = LSTM(100, return_sequences=True)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.2)(x)

        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    @staticmethod
    def create_bilstm_model(
            vocab_size,
            embedding_dim=100,
            max_len=100,
            embedding_matrix=None
    ):
        """Create Bidirectional LSTM model"""
        # Input layer
        inputs = Input(shape=(max_len,))

        # Embedding layer
        if embedding_matrix is not None:
            x = Embedding(
                vocab_size,
                embedding_dim,
                weights=[embedding_matrix],
                trainable=False
            )(inputs)
        else:
            x = Embedding(vocab_size, embedding_dim)(inputs)

        # Bidirectional LSTM
        x = Bidirectional(LSTM(100, return_sequences=True))(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.2)(x)

        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    @staticmethod
    def create_hybrid_model(
            word_vocab_size,
            char_vocab_size,
            embedding_dim=100,
            max_len=100,
            word_embedding_matrix=None
    ):
        """Create hybrid model with word and character embeddings"""
        # Word input
        word_inputs = Input(shape=(max_len,))
        if word_embedding_matrix is not None:
            word_embeddings = Embedding(
                word_vocab_size,
                embedding_dim,
                weights=[word_embedding_matrix],
                trainable=False
            )(word_inputs)
        else:
            word_embeddings = Embedding(
                word_vocab_size,
                embedding_dim
            )(word_inputs)

        word_lstm = LSTM(100, return_sequences=True)(word_embeddings)
        word_pool = GlobalMaxPooling1D()(word_lstm)

        # Character input
        char_inputs = Input(shape=(max_len,))
        char_embeddings = Embedding(char_vocab_size, 50)(char_inputs)
        char_lstm = LSTM(50, return_sequences=True)(char_embeddings)
        char_pool = GlobalMaxPooling1D()(char_lstm)

        # Combine word and character features
        x = Concatenate()([word_pool, char_pool])
        x = Dropout(0.2)(x)

        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[word_inputs, char_inputs], outputs=outputs)
        return model