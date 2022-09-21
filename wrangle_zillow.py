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
    # Changing bedrooms to a discrete variable
    df.bedroomcnt = df.bedroomcnt.astype(object)
    df = df[df.bedroomcnt > 0.0]

    # Changing bedrooms to a discrete variable
    df.bathroomcnt = df.bathroomcnt.astype(object)
    df = df[df.bathroomcnt > 0.0]

    #Changing zipcodes to discrete values
    df.regionidzip = df.regionidzip.astype(object)
    # Dropping rows within the dataframe that are either null or have a lot size of 0.
    df.lotsizesquarefeet = df.lotsizesquarefeet.fillna(0)
    df = df[df.lotsizesquarefeet != 0.0]

    # Dropping duplexes and triplexes
    df.unitcnt = df.unitcnt = 1

    # filtering for only transactions in 2017
    df = df[df.transactiondate < '2018-01-01']

    # Dropping nulls in sqft
    df = df[df['calculatedfinishedsquarefeet'].notna()]

    # create column with fips value converted from an integer to the county name string
    df['fips'] = df.fips.map({6037.0 : 'los_angeles', 6059.0 : 'orange', 6111.0 : 'ventura'})
    
    # Making sure properties are not tax deliquent
    df.taxdelinquencyflag = df.taxdelinquencyflag = 'Y'

    # rename columns for clarity
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms', \
        'calculatedfinishedsquarefeet':'sqft','taxvaluedollarcnt':'tax_value', \
        'lotsizesquarefeet':'lotsize', \
        'fips':'county','regionidzip':'zip'})

    # one-hot encode county
    dummies = pd.get_dummies(df['county'],drop_first=False,dtype=float)
    df = pd.concat([df, dummies], axis=1)

    #dropping columns due to having repeated values
    df = df.drop(columns=['roomcnt','unitcnt','propertylandusedesc','parcelid','id','finishedsquarefeet12','rawcensustractandblock','structuretaxvaluedollarcnt','landtaxvaluedollarcnt'])

    df = df.astype({'bathrooms':'object','bedrooms':'object','calculatedbathnbr':'object','county':'object','fullbathcnt':'object','propertycountylandusecode':'object','propertylandusetypeid':'object','regionidcity':'object','regionidcounty':'object','censustractandblock':'object'})

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

def wrangle_zillow():
    df = get_zillow()
    df = prep_zillow(df)
    df = handle_missing_values(df, .75, .75)

    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    return train, validate, test