import mlflow
import dagshub
import logging
import json

dagshub.init(repo_owner='akshatsharma2407', repo_name='GMC_motors', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/akshatsharma2407/GMC_motors.mlflow')

def load_model_info(file_path : str) -> dict:
    with open(file_path,'r') as f:
        model_info = json.load(f)
    return model_info

def create_uri(run_id,model_path):
    run_id = run_id
    model_path = model_path
    model_uri = f'runs:/{run_id}/{model_path}'
    return model_uri

def register_model(model_uri,model_name):
    model_name = model_name
    model_version = mlflow.register_model(model_uri,model_name).version
    return model_version,model_name

def add_descr_tags(model_name,model_version,description,author_name):
    client = mlflow.tracking.MlflowClient()

    client.update_model_version(
        name=model_name,
        version=model_version,
        description=description
    )

    client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key='author',
        value=author_name
    )

def main():
    model_info = load_model_info('reports/exp_info.json')
    print(model_info)
    model_uri = create_uri(model_info['run_id'],model_info['model_path'])
    model_version,model_name = register_model(model_uri,'GMC_MODEL')
    add_descr_tags(model_name,model_version,'A vew Version of gmc model in registry','akshat')

if __name__ == '__main__':
    main()