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


if __name__ == '__main__':
    unittest.main()