# src/preprocessing/main.py

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.preprocessing.data_processor import DataPreprocessor

if __name__ == "__main__":
    # Define paths relative to project root
    raw_data_path = project_root / 'data' / 'raw' / 'Snappfood.csv'
    processed_data_path = project_root / 'data' / 'processed' / 'processed_snapfood.csv'

    print(f"Processing data from: {raw_data_path}")
    print(f"Output will be saved to: {processed_data_path}")

    # Initialize preprocessor
    preprocessor = DataPreprocessor(raw_data_path)

    try:
        if preprocessor.load_data():
            preprocessor.clean_comments()
            preprocessor.fix_labels()
            stats = preprocessor.get_data_statistics()

            # Save processed data
            preprocessor.save_processed_data(processed_data_path)
            print("\nProcessing completed successfully!")

        else:
            print("\nFailed to load data.")

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback

        traceback.print_exc()
