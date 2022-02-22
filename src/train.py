import pandas as pd
from typing import Tuple
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from feature_engine.encoding import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from config.core import config, DATASET_RAW_DIR, TRAINED_MODEL_DIR
import utils


def prepare_raw_data(df_raw_data: pd.DataFrame) -> pd.DataFrame:
    """Prepares the original data. Creates the target variable Age, by categorizing the variable age.

    Args:
        df_raw_data (pd.DataFrame): DataFrame original com os dados

    Returns:
        pd.DataFrame: Prepared DataFrame
    """
    df_raw_data["age"] = df_raw_data["rings"] + 1.5
    df_raw_data["age"] = pd.cut(
        df_raw_data["age"], bins=[0, 8, 14, max(df_raw_data["age"])], labels=["young", "middle age", "old"]
    )
    df_raw_data.drop("rings", axis=1, inplace=True)

    df_prepared_data = df_raw_data[df_raw_data["height"] != 0]

    return df_prepared_data


def split_train_test(
    df_for_split: pd.DataFrame,
    y_name: str = config.model_config.target,
    test_size: float = config.model_config.test_size,
    stratify_by: str = "sex",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Selects X and y dataframes and splits into train and test

    Args:
        df_for_split (pd.DataFrame): Original dataframe to be splitted
        y_name (str, optional): Name of the target variable. Defaults to config.model_config.target.
        test_size (float, optional): Size of the test dataframe. Defaults to config.model_config.test_size.
        stratify_by (str, optional): Variable name to stratify the split. Defaults to "sex".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: The splitted train and test dataframes. X_train, X_test, y_train, y_test
    """
    X = df_for_split.drop(y_name, axis=1)
    y = df_for_split[y_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config.model_config.random_state, stratify=df_for_split[stratify_by]
    )

    return X_train, X_test, y_train, y_test


def create_dataprep_pipeline() -> Pipeline:
    """Creates the data preparation pipeline

    Returns:
        Pipeline: The data preparation pipeline
    """
    dataprep_pipeline = Pipeline(
        steps=[
            # == CATEGORICAL ENCODING
            ("one_hot_encoder", OneHotEncoder(variables=config.model_config.categorical_vars, drop_last=False)),
            # === SCALLER
            ("minmax_scaller", MinMaxScaler()),
        ]
    )

    return dataprep_pipeline


def fit_pipeline_and_save(X: pd.DataFrame, pipeline: Pipeline, file_name: str) -> Tuple[Pipeline, pd.DataFrame]:
    """Fit, transform and save Pipeline using mlflow

    Args:
        X (pd.DataFrame): DataFrame to fit Pipeline
        pipeline (Pipeline): Pipeline to be fitted and saved
        file_name (str): Name to save Pipeline

    Returns:
        Tuple[Pipeline]: Returns fitted Pipeline
    """
    X_transformed = pipeline.fit(X)
    trained_pipeline = pipeline
    mlflow.sklearn.log_model(trained_pipeline, artifact_path=file_name)

    return trained_pipeline


def grid_search_hyperparameter_tuning(
    model_to_tune: GradientBoostingClassifier,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    search_grid: dict,
    cv: int = 3,
) -> GradientBoostingClassifier:
    """Executes an exhaustive search over specified parameter values (GridSearchCV) for a GradientBoostingClassifier model.

    Args:
        model_to_tune (GradientBoostingClassifier): GradientBoostingClassifier model to be tuned.
        X_train (pd.DataFrame): X DataFrame to fit
        y_train (pd.DataFrame): y DataFrame to fit
        search_grid (dict): Dictionary with parameters names (str) as keys and lists of parameter settings to try.
        cv (int, optional): The number of folds in a (Stratified)KFold. Defaults to 3.

    Returns:
        GradientBoostingClassifier: Tuned GradientBoostingClassifier model.
    """
    model_tuning = GridSearchCV(estimator=model_to_tune, param_grid=search_grid, cv=cv, n_jobs=-1)
    model_tuning.fit(X_train, y_train)
    model_tuned = model_tuning.best_estimator_

    return model_tuned


def evaluate_classification_and_log(
    classification_model: GradientBoostingClassifier,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
):
    """Evaluates a classification model by calculating accuracy and f1 for train and test datsets. Saves the model and metrics using mlflow.

    Args:
        classification_model (GradientBoostingClassifier): Model to be evaluated.
        X_train (pd.DataFrame): X train DataFrame to fit
        y_train (pd.DataFrame): y train DataFrame to fit
        X_test (pd.DataFrame): X test DataFrame to evaluete
        y_test (pd.DataFrame): y test DataFrame to evaluete
    """
    # metrics - train
    y_train_pred = classification_model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred, average="weighted")

    # metrics - test
    y_test_pred = classification_model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred, average="weighted")

    # log metrics
    mlflow.log_metric("accuracy_train", accuracy_train)
    mlflow.log_metric("f1_train", f1_train)
    mlflow.log_metric("accuracy_test", accuracy_test)
    mlflow.log_metric("f1_test", f1_test)

    mlflow.sklearn.log_model(classification_model, "classification_model")
    mlflow.log_param("model_name", type(classification_model).__name__)
    mlflow.log_param("model_params", classification_model.get_params())


if __name__ == "__main__":

    utils.start_experiment(artifact_location=TRAINED_MODEL_DIR, stage="train")

    df_raw_data = utils.load_data(
        dataset_directory=DATASET_RAW_DIR,
        file_name=config.app_config.training_data_file,
        col_names=config.model_config.col_names,
    )

    df_prepared_data = prepare_raw_data(df_raw_data)

    X_train, X_test, y_train, y_test = split_train_test(df_prepared_data)

    dataprep_pipeline = create_dataprep_pipeline()

    trained_dataprep_pipeline = fit_pipeline_and_save(
        X=X_train, pipeline=dataprep_pipeline, file_name=config.app_config.dataprep_pipeline_save_file
    )

    X_train_transformed = pd.DataFrame(trained_dataprep_pipeline.transform(X_train))
    X_test_transformed = pd.DataFrame(trained_dataprep_pipeline.transform(X_test))

    search_grid = {
        "learning_rate": [0.02, 0.05, 0.1],
        "n_estimators": [110, 120, 130],
        "subsample": [0.7, 0.8, 0.85],
        "min_samples_leaf": [6, 7, 8],
        "max_depth": [5, 6, 7],
        "ccp_alpha": [0.0001, 0.0005, 0.001],
    }

    model_to_tune = GradientBoostingClassifier(random_state=123)
    model = grid_search_hyperparameter_tuning(model_to_tune, X_train_transformed, y_train, search_grid=search_grid)
    evaluate_classification_and_log(model, X_train_transformed, y_train, X_test_transformed, y_test)

    mlflow.end_run()
    print(X_train_transformed.describe())