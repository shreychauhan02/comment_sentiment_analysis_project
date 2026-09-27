import numpy as np
import pandas as pd
import logging
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.data.pandas_dataset import PandasDataset
import os
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import lightgbm as lgb
import json
import dagshub
from src.data.utils import Logger, load_params, load_data, load_model
from dotenv import load_dotenv

# Initialize Logger
logger = Logger('model_evaluation', logging.INFO)

def evaluate_model(model: lgb.LGBMClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict, np.ndarray]:
    """
    Evaluate the model and log classification metrics and confusion matrix.

    Parameters
    ----------
    model : lgb.LGBMClassifier
        The trained model to evaluate.
    X_test : np.ndarray
        The test features.
    y_test : np.ndarray
        The true labels for the test data.

    Returns
    -------
    Tuple[Dict, np.ndarray]
        The classification report as a dictionary and the confusion matrix as a numpy array.
    """
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.info('Model evaluation completed')
        return report, cm
    except Exception as e:
        logger.error(f'Error during model evaluation: {str(e)}')
        raise

def log_confusion_matrix(cm: np.ndarray, dataset_name: str) -> None:
    """
    Log confusion matrix as an artifact to MLflow.

    Parameters
    ----------
    cm : np.ndarray
        The confusion matrix as a numpy array.
    dataset_name : str
        The name of the dataset, used to name the artifact.

    Returns
    -------
    None
    """
    try:
        visualizations_dir = Path(__file__).resolve().parents[2] / 'data' / 'visualizations'
        visualizations_dir.mkdir(parents=True, exist_ok=True)
        cm_file_path = visualizations_dir / f'confusion_matrix_{dataset_name}.png'

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(cm_file_path)
        mlflow.log_artifact(cm_file_path)
        plt.clf()  # Clear the figure to free memory
        logger.info(f"Confusion matrix logged for {dataset_name}.")
    except Exception as e:
        logger.error(f"Error logging confusion matrix: {str(e)}")
        raise

def save_model_info(run_id: str, file_path: Path, model_uri: str, vectorizer_uri: str) -> None:
    """
    Save the model run ID and URIs to a JSON file.

    Parameters
    ----------
    run_id : str
        The ID of the MLflow run.
    file_path : Path
        The path to the JSON file to save the model information.
    model_uri : str
        The models:/ URI of the logged LightGBM model.
    vectorizer_uri : str
        The models:/ URI of the logged TF-IDF vectorizer.

    Returns
    -------
    None
    """
    model_info = {
        'run_id': run_id,
        'model_uri': model_uri,
        'vectorizer_uri': vectorizer_uri,
        'model_path':'lgbm_model',
        'vectorizer_path': 'tfidf_vectorizer'
    }
    try:
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.info(f"Model info saved to {file_path}.")
    except Exception as e:
        logger.error(f"Error saving model info: {str(e)}")
        raise


def main() -> None:
    """
    Evaluate a LightGBM model on test data and log metrics and artifacts to MLflow.

    This function logs classification metrics, confusion matrix, and model signature
    to MLflow, and saves model info.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # Initialize Dagshub and set experiment
    if not os.getenv("GITHUB_ACTIONS"):
        load_dotenv()

    dagshub.init(repo_owner="shreychauhan02", repo_name="my_final_project", mlflow=True)
    mlflow.set_experiment('comment-sentiment-dvc')
    
    try:
        with mlflow.start_run(run_name='model_evaluation') as run:
            # Load parameters, model, vectorizer, and test data
            params = load_params(logger)
            root = Path(__file__).resolve().parents[2]
            model_dir = root / 'models'
            test_data_path = root / 'data' / 'interim' / 'test_processed.csv'
            train_data_path = root / 'data' / 'interim' / 'train_processed.csv'
            
            model = load_model(logger, model_dir, 'lgbm_model')
            vectorizer = load_model(logger, model_dir, 'tfidf_vectorizer')
            test_data = load_data(logger, test_data_path)
            train_data = load_data(logger, train_data_path)  # Load train data to log as artifact

            # Log the train and test datasets
            dataset: PandasDataset = mlflow.data.from_pandas(train_data)
            mlflow.log_input(dataset, context="training")
            dataset: PandasDataset = mlflow.data.from_pandas(test_data)  
            mlflow.log_input(dataset, context="testing")

            # Log parameters
            mlflow.log_params(params)

            # Prepare test data
            test_data.dropna(inplace=True)
            test_tfidf = vectorizer.transform(test_data['comment'].values).toarray()
            X_test_tfidf = np.hstack([test_tfidf, test_data[['word_count', 'char_count', 'avg_word_length']].values])
            y_test = test_data['category'].values

            # Log model with signature
            signature = infer_signature(X_test_tfidf, model.predict(X_test_tfidf))
            logged_model = mlflow.sklearn.log_model(model, "lgbm_model", signature=signature, serialization_format="cloudpickle")
            signature = infer_signature(test_data['comment'][0], test_tfidf[0])
            logged_vectorizer = mlflow.sklearn.log_model(vectorizer, "tfidf_vectorizer", signature=signature)

            # Evaluate model
            report, cm = evaluate_model(model, X_test_tfidf, y_test)

            # Log classification report metrics for the test data
            mlflow.log_metric("test_accuracy", float(report['accuracy']))
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            # Log confusion matrix
            log_confusion_matrix(cm, "Test_Data")

            # Save model info
            save_model_info(run.info.run_id, Path(__file__).resolve().parents[2] / 'experiment_info.json',
                            logged_model.model_uri, logged_vectorizer.model_uri)

            # Set experiment tags
            mlflow.set_tag("model", "LightGBM")
            mlflow.set_tag("vectorizer", "TF-IDF")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

    except Exception as e:
        logger.error(f"Failed to complete model evaluation: {e}")
        raise


if __name__ == '__main__':
    main()