# Goal
Building a machine learning model that is able to accurately predict maintaince on water pumps in Tanzania, based on the data of the Tanzanian Ministry of Water

# Project overview
This project contains the following maps:
1. data: containing the data files in csv format
2. eda: contains a notebook that is used to perform exploratory data analysis
3. analyze: contains files for ETL, feature engineering and training & evaluating the classification model
4. images: all images regarding the model evaluation are saved to this folder
5. results: folder to store pickle files containing the predictions

# How to start
Set up conda environment by running: 
`conda create --name <env> --file requirements.txt`

Run the following commands
`python etl.py`,
`python feature_engineering.py` and
`python train_evaluate_model.py`

Additionaly, the predictions are available via a REST API. Run
`python app.py`
and the predictions are visible by performing a GET request
