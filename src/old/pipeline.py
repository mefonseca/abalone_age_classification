from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from feature_engine.encoding import OneHotEncoder

from config.core import config

abalone_pipeline = Pipeline(
    steps=[
        # CATEGORICAL ENCODING
        ("one_hot_encoder", OneHotEncoder(variables=config.model_config.categorical_vars, drop_last=False)),
        # SCALLER
        ("minmax_scaller", MinMaxScaler()),
        # MODEL
        ("GradientBoostingClassifier", GradientBoostingClassifier()),
    ]
)

from pathlib import Path

print(Path().resolve())
