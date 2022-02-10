import pandas as pd
import pathlib
from typing import List, Tuple
import joblib
from sklearn.pipeline import Pipeline


def load_data(dataset_directory: str, file_name: str, col_names: List[str]) -> pd.DataFrame:
    """Loads csv data file

    Args:
        dataset_directory (str): directory where csv file is stored
        file_name (str): file name
        col_names (List[str]): List of column names
    Returns:
        pd.DataFrame: Loaded DataFrame frame
    """

    df_loaded_data = pd.read_csv(
        pathlib.Path(f"{dataset_directory}/{file_name}"),
        header=None,
        names=col_names
    )

    return df_loaded_data

def fit_pipeline_and_save(X: pd.DataFrame, pipeline: Pipeline, directory_to_save_pipeline: str, file_name: str) -> Tuple[Pipeline, pd.DataFrame]:
    """Fit, transform and save Pipeline

    Args:
        X (pd.DataFrame): DataFrame to fit Pipeline
        pipeline (Pipeline): Pipeline to be fitted and saved
        directory_to_save_pipeline (str): Directory to save Pipeline
        file_name (str): File name to save Pipeline

    Returns:
        Tuple[Pipeline, pd.DataFrame]: Returns fitted Pipeline and transformed DataFrame
    """
    X_transformed = pipeline.fit_transform(X)
    trained_pipeline = pipeline
    joblib.dump(trained_pipeline, pathlib.Path(f"{directory_to_save_pipeline}/{file_name}.pkl"))

    return trained_pipeline, X_transformed
