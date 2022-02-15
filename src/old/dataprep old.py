import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from feature_engine.encoding import OneHotEncoder
from feature_engine.transformation import BoxCoxTransformer


def prepare_original_data():
    col_names = [
        "sex",
        "length",
        "diameter",
        "height",
        "whole_weight",
        "shucked_weight",
        "viscera_weight",
        "shell_weight",
        "rings",
    ]
    data = pd.read_csv("../data/raw/abalone_data.txt", header=None, names=col_names)

    data["age"] = data["rings"] + 1.5
    data.drop("rings", axis=1, inplace=True)

    data = data[data["height"] != 0]

    X = data.drop("age", axis=1)
    y = data["age"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=data["sex"])

    return X_train, X_test, y_train, y_test


def train_pipeline():
    CATEGORICAL_VARS = ["sex"]
    NUMERICAL_BOXCOX_VARS = [
        "length",
        "diameter",
        "height",
        "whole_weight",
        "shucked_weight",
        "viscera_weight",
        "shell_weight",
    ]

    abalone_dataprep_pipeline = Pipeline(
        steps=[
            # == CATEGORICAL ENCODING
            ("one_hot_encoder", OneHotEncoder(variables=CATEGORICAL_VARS, drop_last=False)),
            # === SCALLER
            ("minmax_scaller", MinMaxScaler()),
        ]
    )

    return abalone_dataprep_pipeline

    X_train_pipe4 = abalone_dataprep_pipeline4.fit_transform(X_train, y_train)
    X_test_pipe4 = pd.DataFrame(abalone_dataprep_pipeline4.transform(X_test))
