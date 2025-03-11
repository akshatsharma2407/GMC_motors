import mlflow
import dagshub
import logging

dagshub.init(repo_owner='akshatsharma2407', repo_name='GMC_motors', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/akshatsharma2407/GMC_motors.mlflow')

run_id = '67b05a32212b42f0af2bae36b828e09f'
model_path = 'mlflow-artifacts:/7607f355d16d472f8b1283e4f32bb039/5113aa4c43d34275b2a30f03002531a9/artifacts/model'
model_uri = f'runs:/{run_id}/{model_path}'

