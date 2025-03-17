import mlflow.models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import pickle
import json
from sklearn.base import BaseEstimator
import logging
import os
import mlflow
import dagshub
from sklearn.pipeline import Pipeline
import mlflow.sklearn

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

filehandler = logging.FileHandler("errors.log")
filehandler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

console_handler.setFormatter(formatter)
filehandler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(filehandler)


def load_model(path: str) -> BaseEstimator:
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.debug("model.pkl loaded")
        return model
    except FileNotFoundError:
        logger.error("model.pkl not found")
    except Exception as e:
        logger.error("some error occured while loading model.pkl ", e)


def load_pipe(path: str) -> Pipeline:
    try:
        with open(path, "rb") as f:
            pipe = pickle.load(f)
        logger.debug("pipe.pkl loaded")
        return pipe
    except FileNotFoundError:
        logger.error("pipe.pkl not found")
    except Exception as e:
        logger.error("some error occured while loading the pipe", e)


def load_data(path: str) -> pd.DataFrame:
    try:
        test_df = pd.read_csv(path)
        logger.debug("test data loaded successfully")
        return test_df
    except FileNotFoundError:
        logger.error("data file not found")
        raise
    except Exception as e:
        logger.error("some error occured while loading data", e)
        raise


def predict(test_df: pd.DataFrame, model: BaseEstimator) -> dict:
    try:
        xtest = test_df.drop(columns="Price($)")
        ytest = test_df["Price($)"]

        ypred = model.predict(xtest)

        metrics_dict = {
            "mean squared error": mean_squared_error(ytest, ypred),
            "mean_absolute_error": mean_absolute_error(ytest, ypred),
            "r2_score": r2_score(ytest, ypred),
        }
        logger.debug("prediction on test data done !")

        return metrics_dict
    except Exception as e:
        logger.error("some error occured while loading data ", e)
        raise


def save_metrics(path: str, metrics_dict: dict) -> None:
    try:
        with open(path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        logger.debug("metrics saved successfully")
    except Exception as e:
        logger.error("some error occured while saving the evaluation metrics", e)
        raise


def save_model_info(run_id: str, model_name: str, file_path: str) -> None:
    try:
        model_info = {"run_id": run_id, "model_path": model_name}
        with open(file_path, "w") as f:
            json.dump(model_info, f, indent=4)
        logger.debug("model info saved")
    except Exception as e:
        logger.error("some error occured while saving the model info", e)
        raise


def main() -> None:
    try:

        dagshub.init(repo_owner="akshatsharma2407", repo_name="GMC_motors", mlflow=True)
        mlflow.set_tracking_uri(
            "https://dagshub.com/akshatsharma2407/GMC_motors.mlflow"
        )

        mlflow.set_experiment(experiment_name="Exp_for_Production")
        with mlflow.start_run() as run:

            model = load_model("models/RandomForest.pkl")
            test_df = load_data("data/processed/test_processed_df.csv")
            pipe = load_pipe("models/pipe.pkl")
            # loaded only for exp tracking purpose,
            # so that we can directly push the model and pipe to production

            metrics_dict = predict(test_df, model)

            full_pipeline = Pipeline(
                pipe.steps + [("model", model)]
            )  # adding model in pipeline

            save_metrics("reports/metrics.json", metrics_dict)

            data_for_signature = pd.read_csv("GMC_MLOPS/data/raw/GMC_DATA.csv").head(1)
            data_for_signature["AGE OF CAR"] = data_for_signature["AGE OF CAR"].astype(
                str
            )
            data_for_signature["MODEL"] = data_for_signature["MODEL"].astype(str)
            data_for_signature.drop(columns="PRICE($)", inplace=True)
            print(data_for_signature)

            signature = mlflow.models.infer_signature(
                data_for_signature, full_pipeline.predict(data_for_signature)
            )

            mlflow.sklearn.log_model(
                full_pipeline, "transformer+RF_regressor_model", signature=signature
            )
            mlflow.log_metrics(metrics_dict)
            save_model_info(
                run.info.run_id,
                model_name="transformer+RF_regressor_model",
                file_path="reports/exp_info.json",
            )

            logger.debug("main function executed successfully")
    except Exception as e:
        logger.error("some error occured while executing the main function ", e)
        raise


if __name__ == "__main__":
    main()
