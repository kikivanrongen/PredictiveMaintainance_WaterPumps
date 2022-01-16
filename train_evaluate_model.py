import shap
import sqlite3
import logging 

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

logging.basicConfig(filename='log.log', filemode='a', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

def get_train_test_data(df):
    # divide data into feature set and target value
    y = df['status_group']
    X = df.drop(columns=['status_group'])

    # split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    # store train and test data in database
    train_data = pd.concat([X_train, y_train])
    test_data = pd.concat([X_test, y_test])
    train_data.to_sql('water_pumps_train_data', con=con, if_exists='replace')
    test_data.to_sql('water_pumps_test_data', con=con, if_exists='replace')

    return X_train, X_test, y_train, y_test


def train(X_train, X_test, y_train, y_test, model):
    """This functions trains and tests a classifier on data X_train and X_test resp.

    Input arguments:
    X_train: training feature data
    y_train: training target data
    X_test: training feature data
    y_test: training target data
    model: classification model, e.g. XGBClassifier()

    Returns fitted model and array of predictions
    """

    # train model
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    # explain the model's predictions using SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # visualize feature importance
    shap.summary_plot(shap_values, X_test, plot_type="bar", class_names=model.classes_, show=False)
    plt.savefig('images/feature_importance.png')

    return model, y_pred

def evaluate(y_test, y_pred, model):
    """Evaluates model performance by calculating accuracy and confusion matrix

    Input arguments:
    y_test: actual target values
    y_pred: predicted target values
 
    """

    # determine accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # get confusion matrix and plot as heatmap
    cm = confusion_matrix(y_test, y_pred)
    labels = model.classes_
    labels_T = labels.transpose() + ' (pred)'
    sns.heatmap(cm / np.sum(cm), annot=True, xticklabels=labels, yticklabels=labels_T, fmt='.1%', cmap='Blues')
    
    # save figure to image folder
    plt.savefig('images/confusion_matrix.png')

if __name__ == '__main__':
    # connect to database
    con = sqlite3.connect('data/water_pumps.db')

    # retrieve data
    df = pd.read_sql_query('SELECT * from water_pumps_data_processed', con)

    # convert target column to categorical -- necessary for training
    df['status_group'] = df['status_group'].astype('category')

    # split into train and test data
    X_train, X_test, y_train, y_test = get_train_test_data(df)

    # define model
    clf = XGBClassifier(random_state=0)

    # train model
    logging.info("Training model...")
    model, y_pred = train(X_train, X_test, y_train, y_test, clf)

    # determine model performance
    logging.info("Evaluating model...")
    evaluate(y_test, y_pred, model)

    # close connection
    con.close()