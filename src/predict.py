import pandas as pd
from typing import Tuple
import mlflow
import pathlib

from config.core import config, DATASET_RAW_DIR, TRAINED_MODEL_DIR, DATASET_PREDICTED_DIR
import utils


def load_trained_model():
    """Gets the latest tarined experiment run and loads the data preparation pipeline and the classification model.

    Returns:
        Tuple[Pipeline, GradientBoostingClassifier]: Retuns the dataprep Pipeline and the GradientBoostingClassifier model
    """
    runs = mlflow.search_runs(experiment_ids="1", filter_string="params.stage = 'train'")
    runs.dropna(subset=["end_time"], inplace=True)
    latest_artifact_uri = runs["artifact_uri"][0]

    dataprep_pipeline = mlflow.sklearn.load_model(latest_artifact_uri + "/dataprep_pipeline")
    classification_model = mlflow.sklearn.load_model(latest_artifact_uri + "/classification_model")

    return dataprep_pipeline, classification_model


if __name__ == "__main__":

    utils.start_experiment(artifact_location=TRAINED_MODEL_DIR, stage="predict")

    col_names_predict = config.model_config.col_names
    col_names_predict.remove("rings")

    df_predict_data = utils.load_data(
        dataset_directory=DATASET_RAW_DIR,
        file_name=config.app_config.to_predict_data_file,
        col_names=col_names_predict,
    )

    trained_dataprep_pipeline, classification_model = load_trained_model()

    df_predict_data_transformed = pd.DataFrame(trained_dataprep_pipeline.transform(df_predict_data))
    predicted_values = pd.DataFrame(classification_model.predict(df_predict_data_transformed))

    predicted_values.to_csv(
        pathlib.Path(f"{DATASET_PREDICTED_DIR}/{config.app_config.predicted_data_file}"),
        header=None,
        index=None,
        sep=",",
    )

    mlflow.end_run()
