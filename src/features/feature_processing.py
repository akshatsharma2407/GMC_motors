import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import pickle
import yaml
import logging

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel("DEBUG")

stream_handler = logging.StreamHandler()
stream_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("errors.log")
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> float:
    try:
        test_size = yaml.safe_load(open(params_path))["feature_processing"]["test_size"]
        logger.debug("params loaded successfully")
        return test_size
    except FileNotFoundError:
        logger.error("params.yaml file not found while loading params")
        raise
    except Exception as e:
        logger.error("some error occured while loading params", e)
        raise


def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.debug("data loaded successfully")
        return df
    except FileNotFoundError:
        logger.error("data file not found")
        raise
    except Exception as e:
        logger.error("some error occured while loading data", e)
        raise


def splitting_data(
    df: pd.DataFrame, test_size
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        df["AGE OF CAR"] = df["AGE OF CAR"].astype(str)
        df["MODEL"] = df["MODEL"].astype(str)
        xtrain, xtest, ytrain, ytest = train_test_split(
            df.drop(columns=["PRICE($)"]),
            df["PRICE($)"],
            random_state=42,
            test_size=test_size,
        )
        logger.debug("data splitted successfully")
        return xtrain, xtest, ytrain, ytest
    except Exception as e:
        logger.error("some error occured while splitting data", e)
        raise


def ColumnTransformers() -> tuple[ColumnTransformer, ColumnTransformer]:
    try:
        ct1 = ColumnTransformer(
            [
                (
                    "RatingImputer",
                    SimpleImputer(missing_values=-1, strategy="mean"),
                    ["RATING"],
                ),
                (
                    "OHE",
                    ce.TargetEncoder(verbose=1),
                    [
                        "CAR NAME",
                        "MODEL/CLASS",
                        "DEALER NAME",
                        "DEALER LOCATION (CITY)",
                        "DEALER LOCATION (STATE)",
                    ],
                ),
                (
                    "OE",
                    OrdinalEncoder(
                        categories=[
                            [
                                "1937",
                                "1951",
                                "1952",
                                "1966",
                                "1968",
                                "1977",
                                "1979",
                                "1984",
                                "1986",
                                "1987",
                                "1988",
                                "1989",
                                "1996",
                                "1998",
                                "1999",
                                "2000",
                                "2001",
                                "2002",
                                "2003",
                                "2004",
                                "2005",
                                "2006",
                                "2007",
                                "2008",
                                "2009",
                                "2010",
                                "2011",
                                "2012",
                                "2013",
                                "2014",
                                "2015",
                                "2016",
                                "2017",
                                "2018",
                                "2019",
                                "2020",
                                "2021",
                                "2022",
                                "2023",
                                "2024",
                            ],
                            ["Used", "GMC Certified", "New"],
                            [
                                "87",
                                "73",
                                "72",
                                "58",
                                "56",
                                "47",
                                "45",
                                "40",
                                "38",
                                "37",
                                "36",
                                "35",
                                "28",
                                "26",
                                "25",
                                "24",
                                "23",
                                "22",
                                "21",
                                "20",
                                "19",
                                "18",
                                "17",
                                "16",
                                "15",
                                "14",
                                "13",
                                "12",
                                "11",
                                "10",
                                "9",
                                "8",
                                "7",
                                "6",
                                "5",
                                "4",
                                "3",
                                "2",
                                "1",
                                "0",
                            ],
                        ]
                    ),
                    ["MODEL", "STOCK TYPE", "AGE OF CAR"],
                ),
            ],
            remainder="passthrough",
        )
        logger.debug("columntransformer 1 worked successfully")
    except Exception as e:
        logger.error("some error occued while applying columntransfomer 1 ", e)
        raise

    try:
        ct2 = ColumnTransformer([("stdscaler", StandardScaler(), slice(0, 13))])
        logger.debug("columnTransformer 2 worked successfully")
    except Exception as e:
        logger.error("some error occured while applying column transfomer 2 ", e)
        raise

    return ct1, ct2


def CreatingAndExexutingPipeline(
    ct1: ColumnTransformer,
    ct2: ColumnTransformer,
    xtrain: pd.DataFrame,
    xtest: pd.DataFrame,
    ytrain: pd.DataFrame,
    ytest: pd.DataFrame,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    try:
        pipe = Pipeline([("ct1", ct1), ("ct2", ct2)])

        pipe.set_output(transform="pandas")
        xtrain_trans = pipe.fit_transform(xtrain, ytrain)
        xtest_trans = pipe.transform(xtest)
        xtrain_trans["Price($)"] = ytrain
        xtest_trans["Price($)"] = ytest
        logger.debug("creation and execution of pipeline done")
        return (
            pipe,
            xtrain_trans,
            xtest_trans,
        )  # now xtrain_trans is train_df,  xtest_trans is test_df
    except Exception as e:
        logger.error("some error occured while creating and executing pipeline ", e)
        raise


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, path: str) -> None:
    try:
        processed_data_path = os.path.join(path, "processed")
        os.makedirs(processed_data_path, exist_ok=True)
        train_df.to_csv(
            os.path.join(processed_data_path, "train_processed_df.csv"), index=False
        )
        test_df.to_csv(
            os.path.join(processed_data_path, "test_processed_df.csv"), index=False
        )
        logger.debug("data saved successfully")
    except Exception as e:
        logger.error("some error occured while saving data ", e)
        raise


def save_pipeline(pipe: Pipeline, path_to_save: str) -> None:
    try:
        with open(path_to_save, "wb") as f:
            pickle.dump(pipe, f)
        logger.debug("pipe.pkl saved")
    except Exception as e:
        logger.error("some error occued while saving pipe.pkl ", e)
        raise


def main() -> None:
    try:
        test_size = load_params("params.yaml")
        df = load_data("./data/raw/GMC_DATA.csv")
        xtrain, xtest, ytrain, ytest = splitting_data(df, test_size)
        ct1, ct2 = ColumnTransformers()
        pipe, train_df, test_df = CreatingAndExexutingPipeline(
            ct1, ct2, xtrain, xtest, ytrain, ytest
        )
        save_data(train_df, test_df, "./data")
        save_pipeline(pipe, "./models/pipe.pkl")
        logger.debug("main function executed successfully")
    except Exception as e:
        logger.error("some error occured while executing main funcition ", e)
        raise


if __name__ == "__main__":
    main()
