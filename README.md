# Goal
Building a machine learning model that is able to accurately predict maintaince on water pumps in Tanzania, based on the data of the Tanzanian Ministry of Water

# Project overview
This project contains the following maps:
1. *data*: containing the data files in csv format
2. *eda*: contains a notebook that is used to perform exploratory data analysis
3. *analyze*: contains files for ETL, feature engineering and training & evaluating the classification model
4. *images*: all images regarding the model evaluation are saved to this folder
5. *results*: folder to store pickle files containing the predictions

# Getting started (Docker installation)
We launch the Flask application using Docker. Build docker image from root directory with \
`docker build -t <name> .` \
then run the application with \
`docker run -it -d -p 5000:5000 <name>`

# Re-train model
If you are interested in reproducing the results, the following commands can be executed from the analyze folder:
1. Extract, transform and load the data with
`python etl.py`
2. Feature engineering with
`python feature_engineering.py`
3. Train and evaluate the model with
`python train_evaluate_model.py`

The predictions are stored in the results folder, but are also visible through a REST API, by running \
`python app.py` \
(similar to getting started with Docker)
