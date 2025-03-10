import numpy as np
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


def load_data(path : str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def splitting_data(df : pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    df['AGE OF CAR'] = df['AGE OF CAR'].astype(str)
    df['MODEL'] = df['MODEL'].astype(str)
    xtrain,xtest,ytrain,ytest = train_test_split(df.drop(columns=['PRICE($)']),df['PRICE($)'],random_state=42,test_size=0.2)
    return xtrain,xtest,ytrain,ytest

def ColumnTransformers() -> tuple[ColumnTransformer,ColumnTransformer]:
    ct1 = ColumnTransformer(
        [
            ('RatingImputer',SimpleImputer(missing_values=-1,strategy='mean'),['RATING']),
            ('OHE',ce.TargetEncoder(verbose=1),['CAR NAME','MODEL/CLASS','DEALER NAME','DEALER LOCATION (CITY)','DEALER LOCATION (STATE)']),
            ('OE',OrdinalEncoder(categories=
                                [
                                    ["1937", "1951", "1952", "1966", "1968", "1977", "1979", "1984", "1986", "1987", 
            "1988", "1989", "1996", "1998", "1999", "2000", "2001", "2002", "2003", "2004", 
            "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", 
            "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
        ['Used','GMC Certified','New'],
        ["87", "73", "72", "58", "56", "47", "45", "40", "38", "37", "36", "35", "28", 
            "26", "25", "24", "23", "22", "21", "20", "19", "18", "17", "16", "15", "14", 
            "13", "12", "11", "10", "9", "8", "7", "6", "5", "4", "3", "2", "1", "0"]
        ]
        ),['MODEL','STOCK TYPE','AGE OF CAR'])
        ],
        remainder='passthrough'
    )

    ct2 = ColumnTransformer(
        [
            ('stdscaler',StandardScaler(),slice(0,13))
        ]
    )

    return ct1,ct2

def CreatingAndExexutingPipeline(ct1 : ColumnTransformer,ct2 : ColumnTransformer,xtrain : pd.DataFrame,xtest : pd.DataFrame,ytrain : pd.DataFrame,ytest : pd.DataFrame) -> tuple[Pipeline,pd.DataFrame,pd.DataFrame]:
    pipe = Pipeline([
        ('ct1',ct1),
        ('ct2',ct2)
    ])

    pipe.set_output(transform='pandas')
    xtrain_trans = pipe.fit_transform(xtrain,ytrain)
    xtest_trans = pipe.transform(xtest)
    xtrain_trans['Price($)'] = ytrain
    xtest_trans['Price($)'] = ytest
    return pipe,xtrain_trans,xtest_trans #now xtrain_trans is train_df,  xtest_trans is test_df

def save_data(train_df : pd.DataFrame,test_df : pd.DataFrame,path : str) -> None:
    processed_data_path = os.path.join(path,'processed')
    os.makedirs(processed_data_path,exist_ok=True)
    train_df.to_csv(os.path.join(processed_data_path,'train_processed_df.csv'),index=False)
    test_df.to_csv(os.path.join(processed_data_path,'test_processed_df.csv'),index=False)


def save_pipeline(pipe : Pipeline,path_to_save : str) -> None:
    with open(path_to_save, "wb") as f:
        pickle.dump(pipe, f)


def main() -> None:
    df = load_data('data/raw/GMC_DATA.csv')
    xtrain,xtest,ytrain,ytest = splitting_data(df)
    ct1,ct2 = ColumnTransformers()
    pipe,train_df,test_df = CreatingAndExexutingPipeline(ct1,ct2,xtrain,xtest,ytrain,ytest)
    save_data(train_df,test_df,'./data')
    save_pipeline(pipe,'./models/pipe.pkl')

if __name__ == '__main__':
    main()