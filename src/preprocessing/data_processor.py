import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


from src.utils.visualizer import DataVisualizer
from src.preprocessing.normalizer import PersianTextNormalizer

class DataPreprocessor:
    def __init__(self, input_file_path):
        self.input_path = Path(input_file_path)
        # Create plots directory relative to project root
        plots_dir = project_root / 'plots'
        self.visualizer = DataVisualizer(output_dir=plots_dir)
        self.processed_data = None
        self.normalizer = PersianTextNormalizer(
            remove_stopwords=True,
            convert_informal=True
        )
    def load_data(self):
        """Load the data from CSV file with tab delimiter and proper encoding"""
        try:
            # Read with tab delimiter and UTF-8-BOM encoding
            self.data = pd.read_csv(
                self.input_path,
                sep='\t',
                encoding='utf-8-sig',  # handles the BOM character
                quoting=3,  # QUOTE_NONE
                on_bad_lines='warn'
            )

            # Drop any empty columns (sometimes created by trailing tabs)
            self.data = self.data.dropna(axis=1, how='all')

            # Remove header row if it exists
            if 'label' in self.data['label'].values:
                self.data = self.data[self.data['label'] != 'label']

            print(f"Data loaded successfully. Shape: {self.data.shape}")
            print("\nColumns found:", self.data.columns.tolist())
            print("\nUnique labels:", self.data['label'].unique())
            print("Unique label_ids:", self.data['label_id'].unique())
            print("\nFirst few rows:")
            print(self.data.head())
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def clean_comments(self):
        """Clean and normalize comment text"""
        # 1. اول نرمال‌سازی با استفاده از PersianTextNormalizer
        self.data['cleaned_comment'] = self.data['comment'].apply(
            lambda x: self.normalizer.normalize(x)
        )

        # 2. حذف نظرات خالی
        self.data = self.data[self.data['cleaned_comment'].str.len() > 0]

        # 3. حذف کاراکترهای تکراری
        self.data['cleaned_comment'] = self.data['cleaned_comment'].apply(
            self.normalizer.remove_duplicate_chars
        )

        # 4. نرمال‌سازی اعداد
        self.data['cleaned_comment'] = self.data['cleaned_comment'].apply(
            self.normalizer.normalize_numbers
        )

        # 5. پاک‌سازی نهایی متن
        def clean_text(text):
            if pd.isna(text):
                return ""
            text = str(text)
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\u0600-\u06FF\s0-9\.,،!؟]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        self.data['cleaned_comment'] = self.data['cleaned_comment'].apply(clean_text)
        self.data = self.data[self.data['cleaned_comment'].str.len() > 0]


    def fix_labels(self):
        """Fix and validate labels"""
        # Convert label_id to numeric, coercing errors to NaN
        self.data['label_id'] = pd.to_numeric(self.data['label_id'], errors='coerce')

        # Create sentiment labels
        def get_sentiment(row):
            if pd.notna(row['label_id']):
                return 'negative' if row['label_id'] == 1 else 'positive'
            elif pd.notna(row['label']):
                return 'negative' if row['label'] == 'SAD' else 'positive'
            else:
                return np.nan

        self.data['sentiment'] = self.data.apply(get_sentiment, axis=1)

        # Remove rows with invalid labels
        self.data = self.data.dropna(subset=['sentiment'])

        # Verify label consistency
        label_stats = {
            'total': len(self.data),
            'by_sentiment': self.data['sentiment'].value_counts().to_dict(),
            'by_label': self.data['label'].value_counts().to_dict(),
            'by_label_id': self.data['label_id'].value_counts().to_dict()
        }
        print("\nLabel Statistics:")
        print(f"Total samples: {label_stats['total']}")
        print("\nBy sentiment:", label_stats['by_sentiment'])
        print("By label:", label_stats['by_label'])
        print("By label_id:", label_stats['by_label_id'])

    def get_data_statistics(self):
        """Get basic statistics and create visualizations"""
        stats = {
            'total_samples': len(self.data),
            'sentiment_distribution': self.data['sentiment'].value_counts().to_dict(),
            'average_comment_length': self.data['cleaned_comment'].str.len().mean(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'sample_comments': {
                'positive': self.data[self.data['sentiment'] == 'positive']['cleaned_comment'].head(3).tolist(),
                'negative': self.data[self.data['sentiment'] == 'negative']['cleaned_comment'].head(3).tolist()
            }
        }

        # Create visualizations
        self.visualizer.plot_sentiment_distribution(self.data)
        self.visualizer.plot_comment_length_distribution(self.data)
        self.visualizer.plot_word_frequency(self.data)
        self.visualizer.plot_sentiment_by_length(self.data)
        self.visualizer.plot_missing_values(self.data)

        return stats

    def save_processed_data(self, output_path):
        """Save the processed data to CSV"""
        # تبدیل مسیر به Path
        output_path = Path(output_path)

        # اگر مسیر نسبی است، آن را نسبت به ریشه پروژه در نظر بگیریم
        if not output_path.is_absolute():
            output_path = project_root / output_path

        # ایجاد پوشه‌های لازم
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save necessary columns
        save_columns = ['cleaned_comment', 'sentiment', 'label_id']
        self.data[save_columns].to_csv(output_path, index=False, encoding='utf-8')
        print(f"Data saved to {output_path}")

    def train_test_split(self, test_size=0.2, random_state=42):
        """Split the data into training and testing sets"""
        # Ensure we have clean data before splitting
        clean_data = self.data.dropna(subset=['sentiment', 'cleaned_comment'])

        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(
            clean_data,
            test_size=test_size,
            random_state=random_state,
            stratify=clean_data['sentiment']
        )

        return train_data, test_data


if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor('../../data/raw/Snappfood.csv')

    # Load data
    if preprocessor.load_data():
        try:
            # Clean comments
            preprocessor.clean_comments()

            # Fix and validate labels
            preprocessor.fix_labels()

            # Get and print statistics
            stats = preprocessor.get_data_statistics()
            print("\nDataset Statistics:")
            print(f"Total samples: {stats['total_samples']}")

            print("\nSentiment distribution:")
            for sentiment, count in stats['sentiment_distribution'].items():
                print(f"{sentiment}: {count}")

            print(f"\nAverage comment length: {stats['average_comment_length']:.2f} characters")

            print("\nMissing values:")
            for col, count in stats['missing_values'].items():
                if count > 0:  # Only show columns with missing values
                    print(f"{col}: {count}")

            print("\nSample positive comments:")
            for comment in stats['sample_comments']['positive']:
                print(f"- {comment}")

            print("\nSample negative comments:")
            for comment in stats['sample_comments']['negative']:
                print(f"- {comment}")

            # Split data
            train_data, test_data = preprocessor.train_test_split()

            # Save processed data
            preprocessor.save_processed_data('data/processed/processed_snapfood.csv')
            print("\nTrain set shape:", train_data.shape)
            print("Test set shape:", test_data.shape)

        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            import traceback

            traceback.print_exc()
    else:
        print("\nFailed to load data. Please check the file path and format.")