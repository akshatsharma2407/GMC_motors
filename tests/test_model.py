# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import numpy as np

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("AKSHAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "akshatsharma2407"
        repo_name = "GMC_motors"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the new model from MLflow model registry
        cls.new_model_name = "GMC_MODEL"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        cls.pipe = pickle.load(open('models/pipe.pkl', 'rb'))

        # Load holdout test data
        cls.holdout_data = pd.read_csv('data/raw/GMC_DATA.csv').sample(100)

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Create a dummy input for the model based on expected input shape
        # input_text = np.array(['GMC Sierra 3500 Denali','Sierra 3500','2024','New',0,'0',3.1,507.0,'Kunes Chevrolet GMC of Elkhorn','Elkhorn','Wisconsin']).reshape(1,-1)

        columns = [
            "CAR NAME", "MODEL/CLASS", "MODEL", "STOCK TYPE", "MILEAGE", "AGE OF CAR", 
            "RATING", "REVIEW", "DEALER NAME", "DEALER LOCATION (CITY)", "DEALER LOCATION (STATE)"
        ]

        input_text = pd.DataFrame([
            ['GMC Sierra 3500 Denali', 'Sierra 3500', '2024', 'New', 0, '0', 3.1, 507.0, 
            'Kunes Chevrolet GMC of Elkhorn', 'Elkhorn', 'Wisconsin']
        ], columns=columns)

        input_df = self.pipe.transform(input_text)

        # Verify the input shape
        self.assertEqual(input_df.shape[1], len(self.pipe.get_feature_names_out()))

        prediction = self.new_model.predict(input_text)

        # Verify the output shape (assuming regression with a single output)
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)  # Assuming a single output column

    def test_model_performance(self):
        # Extract features and labels from holdout test data
        self.holdout_data['MODEL'] = self.holdout_data['MODEL'].astype(str)
        self.holdout_data['AGE OF CAR'] = self.holdout_data['AGE OF CAR'].astype(str)
        X_holdout = self.holdout_data.drop(columns='PRICE($)')
        y_holdout = self.holdout_data['PRICE($)']

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.80
        expected_precision = 0.80
        expected_recall = 0.80
        expected_f1 = 0.80

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 score should be at least {expected_f1}')

if __name__ == "__main__":
    unittest.main()