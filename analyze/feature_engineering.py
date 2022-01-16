import sqlite3
import pandas as pd
import numpy as np
import logging

from sklearn.preprocessing import LabelEncoder, StandardScaler
from prince import PCA

logging.basicConfig(filename='analyze/log.log', filemode='a', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

def pca(df):
    """
    This function performs Principal Component Analysis on numerical features to reduce the number of dimensions
    """

    # Standardize data
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    scaler.fit(df[numeric_columns])
    df[numeric_columns] = scaler.transform(df[numeric_columns])

    # PCA on numerical data columns
    pca = PCA(n_components=1, n_iter=3)
    water_pumps_pca = pca.fit_transform(df[numeric_columns])
    water_pumps_pca = water_pumps_pca.rename(columns={0: "pca"})

    # add result of PCA to dataframe
    water_pumps_pca.reindex(df.index)
    df = df.join(water_pumps_pca)

    # drop numeric columns -- as they are now combined into one feature column with PCA
    df = df.drop(columns=numeric_columns)

    return df

def encode_features(df):
    """
    This function encodes features -- both label encoding and one hot encoding is used
    """

    # drop columns with too many categories
    df = df.drop(columns=['installer', 'wpt_name', 'subvillage', 'ward'])

    # retrieve column names with categorical data
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # label encoding for ordinal data
    to_label_encode = ['amount_tsh', 'construction_year_group', 'water_quality', 'quantity']
    df[to_label_encode] = df[to_label_encode].apply(LabelEncoder().fit_transform)

    # one hot encoding for non-ordinal data (excl. target column)
    to_hot_encode = set(categorical_columns) - set(to_label_encode) - set(['status_group'])
    df = pd.get_dummies(df, columns=to_hot_encode, drop_first=True)

    return df

if __name__ == '__main__':
    logging.info("Feature engineering...")

    # connect to database
    con = sqlite3.connect('data/water_pumps.db')

    # retrieve data
    df = pd.read_sql_query('SELECT * from water_pumps_data', con)

    # dimensionality reduction with PCA
    df = pca(df)

    # encode categorical features
    df = encode_features(df)

    # store result in table
    df.to_sql('water_pumps_data_processed', index=False, con=con, if_exists='replace')

    # close connection
    con.close()