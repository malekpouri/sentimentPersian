# src/training/trainer.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import sys
import json
from datetime import datetime

from src.models.sentiment_models import SentimentModels

# Add project root to path and setup paths properly
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Define data and model directories relative to project root
DATA_DIR = project_root / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = project_root / 'models'
PLOTS_DIR = project_root / 'plots'

# Create necessary directories
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

from src.preprocessing.vectorizer import TextVectorizer
from src.utils.visualizer import DataVisualizer


class ModelTrainer:
    def __init__(self, config=None):
        """Initialize trainer with configuration"""
        self.config = config or {
            'max_words': 10000,
            'max_len': 100,
            'embedding_dim': 100,
            'batch_size': 32,
            'epochs': 10,
            'model_type': 'lstm'
        }

        self.vectorizer = TextVectorizer(
            max_words=self.config['max_words'],
            max_len=self.config['max_len'],
            embedding_dim=self.config['embedding_dim']
        )

        self.visualizer = DataVisualizer(output_dir=PLOTS_DIR)
        self.model = None
        self.history = None

    def prepare_data(self, data):
        """Prepare data for training"""
        # Process texts
        texts = data['cleaned_comment'].values

        # Convert labels to binary format
        labels = (data['sentiment'] == 'positive').astype(int)

        # Fit and transform texts
        self.vectorizer.fit_word_tokenizer(texts)
        self.vectorizer.fit_char_tokenizer(texts)

        # Train word embeddings
        self.vectorizer.train_word2vec(texts)

        # Prepare word sequences
        word_sequences = self.vectorizer.prepare_data(texts)

        if self.config['model_type'] == 'hybrid':
            char_sequences = self.vectorizer.prepare_data(texts, char_level=True)
            return [word_sequences, char_sequences], labels

        return word_sequences, labels

    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        # First split into train+val and test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Then split train+val into train and val
        val_size_adjusted = val_size / (1 - test_size)

        if isinstance(X, list):  # For hybrid model
            X_train = []
            X_val = []
            for X_part in X_trainval:
                X_train_part, X_val_part, y_train, y_val = train_test_split(
                    X_part, y_trainval, test_size=val_size_adjusted,
                    random_state=42, stratify=y_trainval
                )
                X_train.append(X_train_part)
                X_val.append(X_val_part)
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=val_size_adjusted,
                random_state=42, stratify=y_trainval
            )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def create_model(self):
        """Create the model based on configuration"""
        vocab_size = self.vectorizer.get_vocab_size()
        embedding_matrix = self.vectorizer.create_embedding_matrix()

        if self.config['model_type'] == 'lstm':
            self.model = SentimentModels.create_lstm_model(
                vocab_size,
                self.config['embedding_dim'],
                self.config['max_len'],
                embedding_matrix
            )
        elif self.config['model_type'] == 'bilstm':
            self.model = SentimentModels.create_bilstm_model(
                vocab_size,
                self.config['embedding_dim'],
                self.config['max_len'],
                embedding_matrix
            )
        elif self.config['model_type'] == 'hybrid':
            char_vocab_size = self.vectorizer.get_vocab_size(char_level=True)
            self.model = SentimentModels.create_hybrid_model(
                vocab_size,
                char_vocab_size,
                self.config['embedding_dim'],
                self.config['max_len'],
                embedding_matrix
            )

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def create_callbacks(self, model_dir):
        """Create training callbacks"""
        callbacks = [
            ModelCheckpoint(
                model_dir / 'best_model.keras',  # تغییر پسوند از .h5 به .keras
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=2,
                min_lr=0.00001
            )
        ]
        return callbacks

    def train(self, X_train, y_train, X_val, y_val):
        """Train the model"""
        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=self.create_callbacks(MODELS_DIR)
        )

    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        results = self.model.evaluate(X_test, y_test)
        return dict(zip(self.model.metrics_names, results))

    def save_results(self, results, model_dir):
        """Save training results and config"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = model_dir / f'results_{timestamp}'
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(results_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4)

        # Save metrics
        with open(results_dir / 'metrics.json', 'w') as f:
            json.dump(results, f, indent=4)

        # Plot training history
        self.visualizer.plot_training_history(self.history.history)

    def load_data(self, data_path):
        """Load and prepare the data"""
        try:
            data_path = Path(data_path)
            if not data_path.is_absolute():
                data_path = project_root / data_path

            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found at: {data_path}")

            print(f"Loading data from: {data_path}")
            data = pd.read_csv(data_path)
            print(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def run_training(self, data_path):
        """Run complete training pipeline"""
        print("Loading data...")
        data = self.load_data(data_path)

        print("Preparing data...")
        print("Data shape:", data.shape)
        print("Data columns:", data.columns.tolist())

        # Check if required columns exist
        required_columns = ['cleaned_comment', 'sentiment']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        print("\nPreparing data...")
        X, y = self.prepare_data(data)

        print("Splitting data...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.split_data(X, y)
        print(f"Train set shape: {X_train.shape if not isinstance(X_train, list) else 'Multiple inputs'}")
        print(f"Validation set shape: {X_val.shape if not isinstance(X_val, list) else 'Multiple inputs'}")
        print(f"Test set shape: {X_test.shape if not isinstance(X_test, list) else 'Multiple inputs'}")

        print("\nCreating model...")
        self.create_model()
        print(self.model.summary())

        print("\nTraining model...")
        self.train(X_train, y_train, X_val, y_val)

        print("\nEvaluating model...")
        results = self.evaluate(X_test, y_test)
        print("Test results:", results)

        print("\nSaving results...")
        self.save_results(results, MODELS_DIR)

        return results


if __name__ == "__main__":
    # Configuration
    config = {
        'max_words': 10000,
        'max_len': 100,
        'embedding_dim': 100,
        'batch_size': 32,
        'epochs': 10,
        'model_type': 'lstm'
    }

    # Create trainer
    trainer = ModelTrainer(config)

    try:
        # Run training with absolute path
        data_file = PROCESSED_DATA_DIR / 'processed_snapfood.csv'
        print(f"Looking for data file at: {data_file}")
        results = trainer.run_training(data_file)
        print("Training completed successfully!")
        print("Results:", results)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback

        traceback.print_exc()