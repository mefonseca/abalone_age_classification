from json import load
import pandas as pd
import pathlib
from typing import Tuple
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from feature_engine.encoding import OneHotEncoder

from config.core import config, DATASET_RAW_DIR, TRAINED_MODEL_DIR
import utils

def prepare_raw_data(df_raw_data: pd.DataFrame) -> pd.DataFrame:
    df_raw_data["age"] = df_raw_data["rings"] + 1.5
    df_raw_data["age"] = pd.cut(df_raw_data["age"], bins=[0, 8, 14, max(df_raw_data["age"])], labels=["young", "middle age", "old"])
    df_raw_data.drop("rings", axis=1, inplace=True)

    df_prepared_data = df_raw_data[df_raw_data["height"] != 0]

    return df_prepared_data

def split_train_test(df_for_split: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = df_for_split.drop(config.model_config.target, axis=1)
    y = df_for_split[config.model_config.target]

    X_train, X_test, y_train, y_test = train_test_split(
                                                        X, y, 
                                                        test_size=config.model_config.test_size, 
                                                        random_state=config.model_config.random_state, 
                                                        stratify=df_for_split["sex"]
                                                        )

    return X_train, X_test, y_train, y_test

def create_dataprep_pipeline():
    dataprep_pipeline = Pipeline(steps=[
        # == CATEGORICAL ENCODING
        ('one_hot_encoder', OneHotEncoder(
        variables=config.model_config.categorical_vars, drop_last=False)),
        
        # === SCALLER
        ('minmax_scaller', MinMaxScaler())
    ])

    return dataprep_pipeline



if __name__ == "__main__":

    df_raw_data = utils.load_data(dataset_directory=DATASET_RAW_DIR,
                                  file_name=config.app_config.raw_data_file,
                                  col_names=config.model_config.col_names)
    
    df_prepared_data = prepare_raw_data(df_raw_data)

    X_train, X_test, y_train, y_test = split_train_test(df_prepared_data)

    dataprep_pipeline = create_dataprep_pipeline()

    trained_dataprep_pipeline, X_train_transformed = utils.fit_pipeline_and_save(
                                                                        X=X_train, 
                                                                        pipeline=dataprep_pipeline,
                                                                        directory_to_save_pipeline=TRAINED_MODEL_DIR,
                                                                        file_name=config.app_config.dataprep_pipeline_save_file
    )

    X_test_transformed = pd.DataFrame(trained_dataprep_pipeline.transform(X_test))

