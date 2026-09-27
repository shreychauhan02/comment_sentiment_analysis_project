import numpy as np
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from src.data.utils import Logger, load_params, load_data, save_data, save_model

# Initialize logger
logger = Logger('feature_extraction', logging.INFO)

def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> pd.DataFrame:
    """
    This function applies TF-IDF transformation to the comment column of a given DataFrame containing text data.
    It uses the TfidfVectorizer from scikit-learn to perform the transformation.

    Parameters
    ----------
    train_data : pd.DataFrame
        The DataFrame containing the text data.
    max_features : int
        The maximum number of features to keep.
    ngram_range : tuple
        The range of ngrams to consider.

    Returns
    -------
    pd.DataFrame
        The transformed data with TF-IDF features and the original columns (except 'category').
    """
    try:
        train_data.dropna(inplace=True)

        logger.info("TF-IDF transformation started ...")

        # Perform TF-IDF transformation
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        train_vec = vectorizer.fit_transform(train_data['comment']).toarray()

        # Combine with original features (excluding 'category')
        train_combined = np.hstack([train_vec, train_data[['word_count', 'char_count', 'avg_word_length']].values])

        # Validate shape
        if train_vec.shape[1] + 3 != train_combined.shape[1]:
            raise ValueError('The number of features in the transformed data does not match the expected number')

        # Create a DataFrame with TF-IDF features and other columns
        tfidf_columns = list(vectorizer.get_feature_names_out()) + ['word_count', 'char_count', 'avg_word_length']
        train_tfidf = pd.DataFrame(train_combined, columns=tfidf_columns)

        # Log the completion of TF-IDF transformation
        logger.info(f"TF-IDF transformation complete. Training data shape (without 'category'): {train_tfidf.shape}")

        # Save the vectorizer in the root directory
        vectorizer_path = Path(__file__).resolve().parents[2] / 'models'
        save_model(logger, vectorizer, vectorizer_path, 'tfidf_vectorizer')

        return train_tfidf
    except Exception as e:
        logger.error(f'Error during TF-IDF transformation: {str(e)}')
        raise


def main() -> None:
    """
    Execute the feature extraction pipeline.

    The pipeline involves loading the preprocessed data, applying TF-IDF transformation,
    and saving the transformed data to a given directory.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    try:
        logger.info("Starting feature extraction pipeline...")

        # Load the parameters
        params = load_params(logger)['feature_extraction']

        # Load the data
        data_path = Path(__file__).resolve().parents[2] / 'data'
        train_data = load_data(logger, data_path / 'interim' / 'train_processed.csv')

        # Apply TF-IDF transformation
        train_tfidf = apply_tfidf(train_data, params['max_features'], tuple(params['ngram_range']))

        # Save the transformed data (excluding 'category')
        data_path_processed = data_path / 'processed'
        data_path_processed.mkdir(parents=True, exist_ok=True)

        save_data(logger, train_tfidf, data_path_processed, 'train_tfidf.csv')

        # Save the target column separately
        category_df = train_data[['category']].reset_index(drop=True)
        save_data(logger, category_df, data_path_processed, 'train_target.csv')
        
        logger.info("Feature extraction pipeline completed successfully.")
    except Exception as e:
        logger.error(f'Error during feature extraction:{str(e)}')
        raise


if __name__ == '__main__':
    main()