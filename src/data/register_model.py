import json
import mlflow
import dagshub
import logging
from pathlib import Path
from src.data.utils import Logger
import os
from dotenv import load_dotenv

# Initialize Logger
logger = Logger('register_model', logging.INFO)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file.

    The model info is a dictionary containing the run ID and the model path.

    Parameters
    ----------
    file_path : str
        The path to the JSON file containing the model info.

    Returns
    -------
    dict
        The loaded model info.
    """
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.info(f'Model info loaded from {file_path}')
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict) -> None:
    """
    Register the model to the MLflow Model Registry.

    Register the model by providing the model name and the model info.
    The model info is a dictionary containing the run ID and the model path
    as key-value pairs.

    Parameters
    ----------
    model_name : str
        The name of the model to register.
    model_info : dict
        A dictionary containing the run ID, model path and vectorizer path.

    Returns
    -------
    None
    """
    try:
        # Get the URI (MLflow 3: prefer the logged-model URI; fall back to legacy runs:/ form)
        model_uri = model_info.get('model_uri') or f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        vectorizer_uri = model_info.get('vectorizer_uri') or f"runs:/{model_info['run_id']}/{model_info['vectorizer_path']}"

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name + '-lgbm-model')

        # Register the vectorizer
        vectorizer_version = mlflow.register_model(vectorizer_uri, model_name + '-vectorizer')

        logger.info(f'Model {model_name} version {model_version.version} registered (latest alias points to it automatically).')
        logger.info(f'Vectorizer {model_name} version {vectorizer_version.version} registered.')
    except Exception as e:
        # Log the error message
        logger.error('Error during model registration: %s', e)
        raise

def main() -> None:
    """
    Register the model to the MLflow Model Registry.

    This function is the entry point for the register_model script. It
    initializes the Dagshub client and loads the model info from a JSON file.
    The model is then registered to the MLflow Model Registry.

    If an exception occurs during the model registration process, the
    exception is logged and printed to the console.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    try:
        # Initialize Dagshub
        if not os.getenv("GITHUB_ACTIONS"):
            load_dotenv()

        dagshub.init(repo_owner="shreychauhan02", repo_name="my_final_project", mlflow=True)

        model_info_path = Path(__file__).resolve().parents[2] / 'experiment_info.json'
        model_info = load_model_info(model_info_path)

        # Register the model
        model_name = "sentiment_analysis"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        raise

if __name__ == '__main__':
    main()