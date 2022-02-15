import pandas as pd
import pathlib
from typing import List, Tuple
import joblib
from sklearn.pipeline import Pipeline
import mlflow


def load_data(dataset_directory: str, file_name: str, col_names: List[str]) -> pd.DataFrame:
    """Loads csv data file

    Args:
        dataset_directory (str): directory where csv file is stored
        file_name (str): file name
        col_names (List[str]): List of column names
    Returns:
        pd.DataFrame: Loaded DataFrame frame
    """

    df_loaded_data = pd.read_csv(pathlib.Path(f"{dataset_directory}/{file_name}"), header=None, names=col_names)

    return df_loaded_data


def start_experiment(artifact_location: pathlib.Path, stage: str, experiment_name: str = "abalone_classification"):
    """Creates the mlflow experiment (if it doesn't already exists) and starts the experiment

    Args:
        artifact_location (pathlib.Path): Path where experiment will be saved.
        stage(str): Stage of the data science process. Eg.: train, test, predict
        experiment_name (str, optional): [description]. Defaults to "abalone_classification".
    """
    try:
        mlflow.create_experiment(name=experiment_name, artifact_location=artifact_location)
        print("Creating mlflow experiment")
    except:
        print("Experiment already exists")

    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.log_param("stage", stage)
