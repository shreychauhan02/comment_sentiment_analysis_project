import pandas as pd
import re
import nltk
import logging
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from src.data.utils import Logger, load_data, save_data

# Initialize logger
logger = Logger('data_preprocessing', logging.INFO)


# Transform the data
def preprocess_and_create_features(df: pd.DataFrame) -> None:
    """
    Preprocess comments and create features.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to preprocess.

    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame with features.
    """
    try:
        logger.info("Preprocessing comments and creating features")

        # Preprocess comments (basic cleaning)
        df['comment'] = df['comment'].astype(str)

        df['comment'] = df['comment'].str.strip()

        df.dropna(inplace=True)

        # Lowercase
        df['comment'] = df['comment'].str.lower()

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        df['comment'] = df['comment'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in stop_words])
        )

        # Remove newline characters
        df['comment'] = df['comment'].apply(lambda x: re.sub(r'\n', ' ', x))

        # Feature 1: Word Count
        df['word_count'] = df['comment'].apply(len)

        # Feature 2: Character Count
        df['char_count'] = df['comment'].apply(len)

        # Feature 3: Average Word Length
        df['avg_word_length'] = df['char_count'] / (df['word_count'] + 1)  # +1 to avoid division by zero

        # Remove non-alphanumeric characters, except punctuation
        df['comment'] = df['comment'].apply(
            lambda x: re.sub(r'[^A-Za-z0-9\s!?.,]', '', x)
        )

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        df['comment'] = df['comment'].apply(
            lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()])
        )

        logger.info(f"Preprocessing and feature creation finished with {len(df)} rows and {len(df.columns)} columns.")
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        raise

      
# Main function to execute the processing pipeline
def main() -> None:
    """
    Execute the data processing pipeline.

    Returns
    -------
    None
    """
    try:
        logger.info("Starting data preprocessing pipeline...")

        # Load the data
        data_path = Path(__file__).resolve().parents[2] / "data"
        train_data = load_data(logger, data_path  / "raw" / "train.csv")
        test_data = load_data(logger, data_path  / "raw" / "test.csv")

        # Normalize the text
        preprocess_and_create_features(train_data)
        preprocess_and_create_features(test_data)

        # Create data/interim directory
        data_path = data_path / 'interim'
        data_path.mkdir(parents=True, exist_ok=True)

        # Save the processed data
        save_data(logger, train_data, data_path, "train_processed.csv")
        save_data(logger, test_data, data_path, "test_processed.csv")

        logger.info("Data processing pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Data processing pipeline failed: {str(e)}")
        raise

# Run the script
if __name__ == "__main__":
    nltk.download('wordnet')
    nltk.download('stopwords')
    main()