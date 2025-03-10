from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import pickle
import json

def load_model(path):
    with open(path,'rb') as f:
        model = pickle.load(f)
    return model

def load_data(path):
    test_df = pd.read_csv(path)
    return test_df

def predict(test_df,model):
    xtest = test_df.drop(columns='Price($)')
    ytest = test_df['Price($)']

    ypred = model.predict(xtest)

    metrics_dict = {
    'mean squared error' : mean_squared_error(ytest,ypred),
    'mean_absolute_error':mean_absolute_error(ytest,ypred),
    'r2_score':r2_score(ytest,ypred)
    }
    
    return metrics_dict

def save_metrics(path,metrics_dict):
    with open(path,'w') as f:
        json.dump(metrics_dict,f,indent=4)

def main():
    model = load_model('models/RandomForest.pkl')
    test_df = load_data('data/processed/test_processed_df.csv')
    metrics_dict = predict(test_df,model)
    save_metrics('reports/metrics.json',metrics_dict)

if __name__ == '__main__':
    main()
