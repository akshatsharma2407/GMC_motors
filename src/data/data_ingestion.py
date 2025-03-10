import numpy as np
import pandas as pd
import os


def load_data(path : str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df
    
def preprocessing(df : pd.DataFrame) -> pd.DataFrame:

    df.drop(columns=['PRICE RANGE','MAKE ORIGIN','PARENT COMPANY','IMAGE','BRAND'],inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def save_data(df : pd.DataFrame) -> None:
    raw_data_path = os.path.join('./data','raw')
    os.makedirs(raw_data_path,exist_ok=True)
    df.to_csv(os.path.join(raw_data_path,'GMC_DATA.csv'),index=False)

def main() -> None:
    df = load_data('C:/Users/aksha/Downloads/CLEANED_GMC_DIESEL.csv')
    df_cleaned = preprocessing(df)
    save_data(df_cleaned)

if __name__ == '__main__':
    main()