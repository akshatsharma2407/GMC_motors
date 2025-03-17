from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle
from sklearn.base import BaseEstimator
import yaml
import logging
import os

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("errors.log")
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(path: str):
    try:
        params = yaml.safe_load(open(path))["model_building"]
        logger.debug("params loaded successfully")
        return params
    except FileNotFoundError:
        logger.error("params.yaml file not found")
        raise
    except Exception as e:
        logger.error("some error occured while loading params", e)
        raise


def load_data(path: str) -> pd.DataFrame:
    try:
        train_data = pd.read_csv(path)
        logger.debug("data loaded successfully")
        return train_data
    except FileNotFoundError:
        logger.error("file not found while loading data")
        raise
    except Exception as e:
        logger.error("some error occured while loading data", e)
        raise


def trainingRandomForest(train_data: pd.DataFrame, params) -> BaseEstimator:
    try:
        xtrain = train_data.drop(columns=["Price($)"])
        ytrain = train_data["Price($)"]
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            bootstrap=params["bootstrap"],
            n_jobs=-1,
            random_state=42,
        )
        model.fit(xtrain, ytrain)
        logger.debug("RandomForest Model trained successfully")
        return model
    except Exception as e:
        logger.error("some error occured while running RandomForest", e)
        raise


def save_model(model: BaseEstimator, path: str) -> None:
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.debug("model saved successfully")
    except Exception as e:
        logger.critical("some error occured while saving model ", e)
        raise


def main() -> None:
    try:
        params = load_params("params.yaml")
        train_data = load_data("./data/processed/train_processed_df.csv")
        model = trainingRandomForest(train_data, params)
        save_model(model, "models/RandomForest.pkl")
        logger.debug("main function executed")
    except Exception as e:
        logger.critical("some error occured while executing main function", e)
        raise


if __name__ == "__main__":
    main()
