from flask import Flask,render_template,request
import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import pandas as pd
import os

app = Flask(__name__)

dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("dagshub token environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri("https://dagshub.com/akshatsharma2407/GMC_motors.mlflow")

# load model from model registry
def get_latest_model_version(model_name):
    client = MlflowClient()
    latest_version = client.get_latest_versions(model_name,stages=['Production'])

    if not latest_version: 
        raise ValueError(f"No model version found for {model_name} in 'Production' stage.")
    
    return latest_version[0].version

model_name =  'GMC_MODEL'
version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{version}"
model = mlflow.pyfunc.load_model(model_uri)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
        input_data = {
            'CAR NAME': [request.form['car_name']],
            'MODEL/CLASS': [request.form['model_class']],
            'MODEL': [request.form['model']],
            'STOCK TYPE': [request.form['stock_type']],
            'MILEAGE': [int(request.form['mileage'])],
            'AGE OF CAR': [request.form['age_of_car']],
            'RATING': [float(request.form['rating'])],
            'REVIEW': [float(request.form['review'])],
            'DEALER NAME': [request.form['dealer_name']],
            'DEALER LOCATION (CITY)': [request.form['dealer_city']],
            'DEALER LOCATION (STATE)': [request.form['dealer_state']]
        }

        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)

        # Make prediction
        prediction = model.predict(input_df)

        return str(int(prediction[0]))

app.run(debug=True,host='0.0.0.0')