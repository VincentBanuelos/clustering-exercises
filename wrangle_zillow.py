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

def get_zillow():
    '''
    Pulls data from codeup database and writes into a dataframe
    '''
    if os.path.isfile('zillow.csv'):
        return pd.read_csv('zillow.csv', index_col=0)
    
    else:

        url = f"mysql+pymysql://{user}:{password}@{host}/zillow"
        
        query = '''
                SELECT
                prop.*,
                predictions_2017.logerror,
                predictions_2017.transactiondate,
                air.airconditioningdesc,
                arch.architecturalstyledesc,
                build.buildingclassdesc,
                heat.heatingorsystemdesc,
                landuse.propertylandusedesc,
                story.storydesc,
                construct.typeconstructiondesc
                FROM properties_2017 prop
                JOIN (SELECT parcelid, MAX(transactiondate) AS max_transactiondate
                FROM predictions_2017
                GROUP BY parcelid) pred USING(parcelid)
                JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid AND pred.max_transactiondate = predictions_2017.transactiondate
                LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
                LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
                LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
                LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
                LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
                LEFT JOIN storytype story USING (storytypeid)
                LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
                WHERE prop.latitude IS NOT NULL
                AND prop.longitude IS NOT NULL
                AND transactiondate <= '2017-12-31'
                AND propertylandusedesc = "Single Family Residential"
                '''

        df = pd.read_sql(query, url)

        df.to_csv('zillow.csv')

    return df


def nulls_by_col(df):
    '''
    This function will return a dataframe that shows how many 
    values in a column are null and what percentage of the columns those nulls make
    '''
    num_missing = df.isnull().sum()
    percnt_miss = num_missing / df.shape[0] * 100
    cols_missing = pd.DataFrame(
    {
        'num_rows_missing': num_missing,
        'percent_rows_missing': percnt_miss
    })
    return  cols_missing


def prep_zillow(df):
    # Dropping rows within the dataframe that are either null or have a lot size of 0.
    df.lotsizesquarefeet = df.lotsizesquarefeet.fillna(0)
    df.lotsizesquarefeet = df.lotsizesquarefeet != 0

    # Dropping duplexes and triplexes
    df.unitcnt = df.unitcnt = 1

    return df


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