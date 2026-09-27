import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from pathlib import Path
import logging
from src.data.utils import Logger, load_params, load_data, save_data

logger = Logger('data_ingestion', logging.INFO)


# Function to preprocess the dataset
def preprocess_data(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Preprocess the dataset by dropping columns and encoding target variables.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to preprocess.
    target_column : str
        The column containing the target variable.

    Returns
    -------
    pd.DataFrame
        The preprocessed dataset.
    """
    try:
        logger.info("Preprocessing data")
        # Removing missing values
        df.dropna(inplace=True)

        # Removing duplicates
        df.drop_duplicates(inplace=True)

        # Removing rows with empty strings
        df = df[df['clean_comment'].str.strip() != '']

        df.rename(columns={'clean_comment': 'comment'}, inplace=True)

        # Mapping target values
        if target_column in df.columns:
            df.loc[:, target_column] = df[target_column].map({-1: 0, 0: 1, 1: 2})
            logger.debug(f"Encoded target variable '{target_column}' with classes: {df[target_column].unique()}")
        else:
            logger.critical(f"Target variable '{target_column}' not found in the dataframe.")
            raise KeyError(f"Missing column in the dataframe: {target_column}")
        
        logger.info('Data preprocessing completed: Missing values, duplicates, and empty strings removed.')
        return df     
    except KeyError as e:
        logger.error(f"Error in preprocessing: {e}")
        raise e
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

# Function to split the dataset into train, validation and test sets
def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, validation, and test sets with stratification on the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset containing both features and the target column.
    target_column : str
        The name of the target column for stratification.
    test_size : float
        The proportion of the dataset to include in the test set.
    random_state : int
        The random state to set for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The train and test sets, each containing both features and target data.
    """
    try:
        logger.info(f"Splitting data with test size {test_size}")

        # Separate features and target variable
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split data into training and test sets with stratification
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size, random_state=random_state, stratify=y)
        
        # Combine X and y back into DataFrames for each split
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        logger.info(f"Data split completed: {len(train_data)} train records and {len(test_data)} test records.")
        return train_data, test_data

    except Exception as e:
        logger.error(f"Error during data splitting: {str(e)}")
        raise e

# Main function to execute the pipeline
def main() -> None:
    """
    Execute the data ingestion pipeline.

    The pipeline involves loading parameters, loading the data, preprocessing the data,
    splitting the data into train and test sets, and saving the datasets to a given directory.

    Parameters
    ----------
    params_path : str
        The path to the YAML file containing the parameters for the pipeline.

    Returns
    -------
    None
    """
    try:
        logger.info("Starting data ingestion pipeline...")
        # Load parameters
        params = load_params(logger)['data_ingestion']

        # Load the data
        url = 'https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv'
        df = load_data(logger, url)

        # Preprocess the data
        target_column = 'category'
        
        final_df = preprocess_data(df, target_column)

        # Split the data into train and test sets
        test_size = float(params['test_size'])
        random_state = params['random_state']
        train_data, test_data = split_data(final_df, target_column, test_size, random_state)

        # Save the datasets
        data_path = Path(__file__).resolve().parents[2] / "data" / "raw"
        save_data(logger, train_data, data_path, "train.csv")
        save_data(logger, test_data, data_path, "test.csv")

        logger.info("Data ingestion pipeline completed successfully.")

    except Exception as e:
        logger.critical(f"Data ingestion pipeline failed: {str(e)}")
        raise e

# Run the script
if __name__ == "__main__":
    main()