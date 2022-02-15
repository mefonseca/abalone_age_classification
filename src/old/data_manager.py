import pandas as pd
import src.config.core as core  # import DATASET_RAW_DIR, TRAINED_MODEL_DIR, config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    data = pd.read_csv(Path(f"{DATASET_RAW_DIR}/{file_name}"), header=None, names=config.model_config.col_names)

    # Creating the Age variable for classification:
    data["age"] = data["rings"] + 1.5
    data["age"] = pd.cut(data["age"], bins=[0, 8, 14, max(data["age"])], labels=["young", "middle age", "old"])
    data.drop("rings", axis=1, inplace=True)

    return data


# load_dataset(config.app_config.raw_data_file)
