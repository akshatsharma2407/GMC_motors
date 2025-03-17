import mlflow
import logging
import json
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


dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("dagshub token not found in model_evaluation.py")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri("https://dagshub.com/akshatsharma2407/GMC_motors.mlflow")


def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, "r") as f:
            model_info = json.load(f)
            logger.debug("model info laoded successfully")
        return model_info
    except FileNotFoundError:
        logger.error("model_info.json is not found")
    except Exception as e:
        logger.error("some error occured whilie loading model info", e)
        raise


def create_uri(run_id, model_path):
    try:
        run_id = run_id
        model_path = model_path
        model_uri = f"runs:/{run_id}/{model_path}"
        logger.debug("uri created successfully")
        return model_uri
    except Exception as e:
        logger.error("some error occured while loading model info", e)
        raise


def register_model(model_uri, model_name):
    try:
        model_name = model_name
        model_version = mlflow.register_model(model_uri, model_name).version
        return model_version, model_name
    except Exception as e:
        logger.error("some error occured while registering a model", e)
        raise


def add_descr_tags(model_name, model_version, description, author_name):
    try:
        client = mlflow.tracking.MlflowClient()

        client.transition_model_version_stage(
            name=model_name, version=model_version, stage="Staging"
        )

        client.update_model_version(
            name=model_name, version=model_version, description=description
        )

        client.set_model_version_tag(
            name=model_name, version=model_version, key="author", value=author_name
        )
    except Exception as e:
        logger.error("some error occured while adding descr tags", e)
        raise


def main():
    try:
        model_info = load_model_info("reports/exp_info.json")
        print(model_info, "=" * 500)
        model_uri = create_uri(model_info["run_id"], model_info["model_path"])
        model_version, model_name = register_model(model_uri, "GMC_MODEL")
        add_descr_tags(
            model_name, model_version, f"V{model_version} GMC model added", "akshat"
        )
        logger.debug("main function executed!!")
    except Exception as e:
        logger.critical("some error occured while exe the main func", e)
        raise


if __name__ == "__main__":
    main()
