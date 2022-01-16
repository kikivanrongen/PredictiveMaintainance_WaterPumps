import numpy as np
import pandas as pd
import sqlite3
import logging 

logging.basicConfig(filename='analyze/log.log', filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

def extract_data():
    """
    This function extracts the data from the provided csv files into dataframes
    """

    # load data
    water_pumps = pd.read_csv('data/water_pump_set.csv', index_col='id', parse_dates=['date_recorded'])
    water_pumps_labels = pd.read_csv('data/water_pump_labels.csv', index_col='id', dtype={'status_group': 'category'})

    # join dataframes on index
    df = water_pumps.join(water_pumps_labels)

    return df

def clean_data(df):
    """
    This function cleans the data 
    """

    # drop unuseful columns
    df = df.drop(columns=['funder', 'region', 'num_private', 'recorded_by', 'scheme_name', 'extraction_type_group', 'management_group', 'payment', 'quantity_group', 'quality_group', 'source_type', 'waterpoint_type_group'])

    # set incorrect values to nan
    df['longitude'] = df['longitude'].replace(0, np.nan)

    # extract 'year' from date recorded
    df['year_recorded'] = df['date_recorded'].dt.year.astype('category')
    df = df.drop(columns=['date_recorded'])

    # convert some numeric columns to categorical
    to_categorical = ['district_code', 'region_code']
    df[to_categorical] = df[to_categorical].astype('category')

    # set remaining (non-numeric) columns to categorical
    to_categorical = df.select_dtypes(include='object').columns.tolist()
    df[to_categorical] = df[to_categorical].astype('category')

    return df

def missing_values_treatment(df):
    """
    This function alters data based on detected missing values 
    """

    # create bins for column with too many missing valyes (or drop?)
    df['construction_year_group'] = pd.cut(x=df['construction_year'], 
        bins=[-1, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020], 
        labels=['Unknown', '1960s', '1970s', '1980s', '1090s', '2000s', '2010s', '2020s'])

    # drop columns with too many missing values (+/- 50%)
    df = df.drop(columns=['construction_year'])

    # fill na's of numerical columns with mean
    df['longitude'] = df['longitude'].fillna(df['longitude'].mean())

    # fill na's of categorical columns with mode
    categorical_columns = df.select_dtypes(include='category').columns.tolist()
    for col in categorical_columns:
        df[col].fillna(df[col].value_counts().idxmax(), inplace=True)

    # drop rows with na's
    df = df.dropna()

    return df

def outlier_treatment(df):
    """
    This function alters data based on detected outliers 
    """

    # log transformation on skewed data so it more resembles normal distribution
    df['population'] = (df['population']+1).transform(np.log)

    # bin features with clear distinctive distribution
    df['amount_tsh'] = pd.cut(x=df['amount_tsh'], 
        bins=[-1, 5, np.inf], 
        labels=['Low', 'High'])
    
    return df

if __name__ == '__main__':
    logging.info("Data preprocessing...")

    # extract data from csv files
    df = extract_data()

    # transform data
    df = clean_data(df)
    df = missing_values_treatment(df)
    df = outlier_treatment(df)

    # connect to database
    con = sqlite3.connect('data/water_pumps.db')
    
    # store in database
    df.to_sql('water_pumps_data', index=False, con=con, if_exists='replace')

    # close connection 
    con.close()

