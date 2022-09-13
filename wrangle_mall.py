import os
import pandas as pd
import numpy as np

from env import user, password, host

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings("ignore")


def get_mall():
    '''
    Pulls data from codeup database and writes into a dataframe
    '''
    url = f"mysql+pymysql://{user}:{password}@{host}/mall_customers"
    
    query = '''
            SELECT * FROM customers
            '''

    df = pd.read_sql(query, url)

    return df

def my_train_test_split(df, target):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    return train, validate, test


def handle_missing_values(df, prop_required_column, prop_required_row):
    
    prop_null_column = 1 - prop_required_column
    
    for col in list(df.columns):
        
        null_sum = df[col].isna().sum()
        null_pct = null_sum / df.shape[0]
        
        if null_pct > prop_null_column:
            df.drop(columns=col, inplace=True)
            
    threshold = int(prop_required_row * df.shape[1])
    
    df.dropna(axis=0, thresh=threshold, inplace=True)
    
    return df