import numpy as np
import pandas as pd
import os


def load_data(path):
    df = pd.read_csv(path)
    return df
    
def preprocessing(df):

    df.drop(columns=['PRICE RANGE','MAKE ORIGIN','PARENT COMPANY','IMAGE','BRAND'],inplace=True)
    df['AGE OF CAR'] = df['AGE OF CAR'].astype(str)
    df['MODEL'] = df['MODEL'].astype(str)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def save_data(df):
    raw_data_path = os.path.join('./data','raw')
    os.makedirs(raw_data_path,exist_ok=True)
    df.to_csv(os.path.join(raw_data_path,'GMC_DATA.csv'))

def main():
    df = load_data('C:/Users/aksha/Downloads/CLEANED_GMC_DIESEL.csv')
    df_cleaned = preprocessing(df)
    save_data(df_cleaned)

if __name__ == '__main__':
    main()