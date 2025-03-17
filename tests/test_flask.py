import unittest
from flask_app.app import app
import pandas as pd

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Car Details Form</title>', response.data)

    def test_predict_car_price(self):
        # Define input data matching the expected model format
        columns = [
            "CAR NAME", "MODEL/CLASS", "MODEL", "STOCK TYPE", "MILEAGE", "AGE OF CAR",
            "RATING", "REVIEW", "DEALER NAME", "DEALER LOCATION (CITY)", "DEALER LOCATION (STATE)"
        ]
        
        input_text = pd.DataFrame([
            ['GMC Sierra 3500 Denali', 'Sierra 3500', '2024', 'New', 0, '0', 3.1, 507.0,
            'Kunes Chevrolet GMC of Elkhorn', 'Elkhorn', 'Wisconsin']
        ], columns=columns)

        # Send a POST request to the prediction endpoint
        response = self.client.post('/predict', json=input_text.to_dict(orient='records'))

        # Check if the response is successful (HTTP 200 OK)
        self.assertEqual(response.status_code, 200)

        # Ensure that the response contains a valid predicted price (assuming numeric output)
        self.assertIn("PRICE($)", response.json, "Response should contain 'PRICE($)' key")
        self.assertIsInstance(response.json["PRICE($)"], (int, float), "Predicted price should be a number")


if __name__ == '__main__':
    unittest.main()