from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import pickle
from sklearn.base import BaseEstimator

def load_data(path : str) -> pd.DataFrame:
    train_data = pd.read_csv(path)
    return train_data

def trainingRandomForest(train_data : pd.DataFrame) -> BaseEstimator:
    xtrain = train_data.drop(columns=['Price($)'])
    ytrain = train_data['Price($)']
    model = RandomForestRegressor()
    model.fit(xtrain,ytrain)
    return model

def save_model(model : BaseEstimator,path : str) -> None:
    with open(path,'wb') as f:
        pickle.dump(model,f)

def main() -> None:
    train_data = load_data('data/processed/train_processed_df.csv')
    model = trainingRandomForest(train_data)
    save_model(model,'models/RandomForest.pkl')

if __name__ == '__main__':
    main()