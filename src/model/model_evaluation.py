from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.base import BaseEstimator
import logging
import os 

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

filehandler = logging.FileHandler('errors.log')
filehandler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
filehandler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(filehandler)

def load_model(path : str) -> BaseEstimator:
    try:
        with open(path,'rb') as f:
            model = pickle.load(f)
        logger.debug('model.pkl loaded')
        return model
    except FileNotFoundError:
        logger.error('model.pkl not found')
    except Exception as e:
        logger.error('some error occured while loading model.pkl ',e)

def load_data(path : str) -> pd.DataFrame:
    try:
        test_df = pd.read_csv(path)
        logger.debug('test data loaded successfully')
        return test_df
    except FileNotFoundError:
        logger.error('data file not found')
        raise
    except Exception as e:
        logger.error('some error occured while loading data', e)
        raise

def predict(test_df : pd.DataFrame,model : BaseEstimator) -> dict:
    try:
        xtest = test_df.drop(columns='Price($)')
        ytest = test_df['Price($)']

        ypred = model.predict(xtest)

        metrics_dict = {
        'mean squared error' : mean_squared_error(ytest,ypred),
        'mean_absolute_error':mean_absolute_error(ytest,ypred),
        'r2_score':r2_score(ytest,ypred)
        }
        logger.debug('prediction on test data done !')
        
        return metrics_dict
    except Exception as e:
        logger.error('some error occured while loading data ',e) 
        raise

def save_metrics(path : str,metrics_dict : dict) -> None:
    try:
        with open(path,'w') as f:
            json.dump(metrics_dict,f,indent=4)
        logger.debug('metrics saved successfully')
    except Exception as e:
        logger.error('some error occured while saving the evaluation metrics', e)
        raise

def main() -> None:
    try:
        model = load_model('models/RandomForest.pkl')
        test_df = load_data('data/processed/test_processed_df.csv')
        metrics_dict = predict(test_df,model)
        save_metrics('reports/metrics.json',metrics_dict)
        logger.debug('main function executed successfully')
    except Exception as e:
        logger.error('some error occured while executing the main function ',e)
        raise


if __name__ == '__main__':
    main()