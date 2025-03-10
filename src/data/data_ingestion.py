import numpy as np
import pandas as pd
import os
import logging

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(path : str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.debug('data loaded successfully')
        return df
    except FileNotFoundError:
        logger.error('file not found while loading data')
        raise
    except Exception as e:
        logger.error('Some error occured while loading the data ',e)
        raise
    
def preprocessing(df : pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['PRICE RANGE','MAKE ORIGIN','PARENT COMPANY','IMAGE','BRAND'],inplace=True)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        logger.debug('preprocessing done successfullty')
        return df
    except Exception as e:
        logger.error('Some error occured while preprocessing ,', e)
        raise

def save_data(df : pd.DataFrame) -> None:
    try:
        raw_data_path = os.path.join('./data','raw')
        os.makedirs(raw_data_path,exist_ok=True)
        df.to_csv(os.path.join(raw_data_path,'GMC_DATA.csv'),index=False)
        logger.debug('data saved successfully')
    except Exception as e:
        logger.critical('Some error occured while saving data ,',e)
        raise

def main() -> None:
    try:
        df = load_data('C:/Users/aksha/Downloads/CLEANED_GMC_DIESEL.csv')
        df_cleaned = preprocessing(df)
        save_data(df_cleaned)
        logger.debug('main funciton executed successfully')
    except Exception as e:
        logger.critical('some error occured in main funciton', e)
        raise

if __name__ == '__main__':
    main()