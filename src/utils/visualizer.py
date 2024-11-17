# src/utils/visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import arabic_reshaper
from bidi.algorithm import get_display


class DataVisualizer:
    def __init__(self, output_dir='plots'):
        """
        Initialize the visualizer with an output directory
        Args:
            output_dir (str): Directory to save the plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style for all plots
        plt.style.use('seaborn-v0_8')
        sns.set_theme()

        # Set font for Persian text
        plt.rcParams['font.family'] = 'DejaVu Sans'  # فونتی که از فارسی پشتیبانی می‌کند

    def reshape_persian_text(self, text):
        """Reshape Persian text for proper display"""
        reshaped_text = arabic_reshaper.reshape(text)
        return get_display(reshaped_text)

    def save_plot(self, plt, name):
        """Save plot with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        plt.savefig(self.output_dir / filename, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_sentiment_distribution(self, data, save=True):
        """Plot sentiment distribution"""
        plt.figure(figsize=(10, 6))
        sns.countplot(data=data, x='sentiment')

        # Set Persian labels with proper reshaping
        plt.title(self.reshape_persian_text('توزیع نظرات مثبت و منفی'))
        plt.xlabel(self.reshape_persian_text('احساس'))
        plt.ylabel(self.reshape_persian_text('تعداد'))

        # Add value labels on top of bars
        for p in plt.gca().patches:
            plt.gca().annotate(f'{int(p.get_height())}',
                               (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='center',
                               xytext=(0, 10), textcoords='offset points')

        if save:
            self.save_plot(plt, 'sentiment_distribution')
        else:
            plt.show()

    def plot_comment_length_distribution(self, data, save=True):
        """Plot comment length distribution"""
        plt.figure(figsize=(12, 6))
        comment_lengths = data['cleaned_comment'].str.len()

        sns.histplot(comment_lengths, bins=50)
        plt.title(self.reshape_persian_text('توزیع طول نظرات'))
        plt.xlabel(self.reshape_persian_text('طول نظر'))
        plt.ylabel(self.reshape_persian_text('تعداد'))

        if save:
            self.save_plot(plt, 'comment_length_distribution')
        else:
            plt.show()

    def plot_word_frequency(self, data, top_n=20, save=True):
        """Plot most frequent words"""
        plt.figure(figsize=(15, 8))

        # Count words
        all_words = ' '.join(data['cleaned_comment']).split()
        word_freq = pd.Series(all_words).value_counts()

        # Reshape Persian words
        reshaped_words = [self.reshape_persian_text(word) for word in word_freq.head(top_n).index]

        # Plot top N words
        sns.barplot(x=word_freq.head(top_n).values,
                    y=reshaped_words)
        plt.title(self.reshape_persian_text(f'{top_n} کلمه پرتکرار'))
        plt.xlabel(self.reshape_persian_text('تعداد تکرار'))
        plt.ylabel(self.reshape_persian_text('کلمه'))

        if save:
            self.save_plot(plt, 'word_frequency')
        else:
            plt.show()

    def plot_sentiment_by_length(self, data, save=True):
        """Plot sentiment distribution by comment length"""
        plt.figure(figsize=(12, 6))

        data['comment_length'] = data['cleaned_comment'].str.len()
        sns.boxplot(x='sentiment', y='comment_length', data=data)
        plt.title(self.reshape_persian_text('توزیع طول نظرات بر اساس احساس'))
        plt.xlabel(self.reshape_persian_text('احساس'))
        plt.ylabel(self.reshape_persian_text('طول نظر'))

        if save:
            self.save_plot(plt, 'sentiment_by_length')
        else:
            plt.show()

    def plot_missing_values(self, data, save=True):
        """Plot missing values heatmap"""
        plt.figure(figsize=(10, 6))

        missing_data = data.isnull().sum()
        missing_df = pd.DataFrame({
            'column': missing_data.index,
            'missing_count': missing_data.values
        })

        sns.barplot(x='column', y='missing_count', data=missing_df)
        plt.title(self.reshape_persian_text('مقادیر گمشده در هر ستون'))
        plt.xticks(rotation=45)
        plt.xlabel(self.reshape_persian_text('ستون'))
        plt.ylabel(self.reshape_persian_text('تعداد مقادیر گمشده'))

        if save:
            self.save_plot(plt, 'missing_values')
        else:
            plt.show()

    def plot_training_history(self, history, metrics=['accuracy', 'loss'], save=True):
        """Plot training history"""
        plt.figure(figsize=(12, 4))

        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, len(metrics), i)
            plt.plot(history[metric], label='Train')
            plt.plot(history[f'val_{metric}'], label='Validation')
            plt.title(self.reshape_persian_text(f'Model {metric}'))
            plt.xlabel(self.reshape_persian_text('Epoch'))
            plt.ylabel(self.reshape_persian_text(metric))
            plt.legend()

        plt.tight_layout()
        if save:
            self.save_plot(plt, 'training_history')
        else:
            plt.show()