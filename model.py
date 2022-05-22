"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#load the data
train_data = pd.read_csv('df_train.csv')
test_data = pd.read_csv('df_test.csv')

def _preprocess_data(data):
    data=[train_data, test_data]
df=pd.concat(data)

#something should be done with the time column
df['time'] = pd.to_datetime(df['time'])

# extract relevant metrics
df['Year'] = df['time'].dt.year
df['Month'] = df['time'].dt.month
df['Weekday'] = df['time'].dt.dayofweek
df['Hour'] = df['time'].dt.hour

#after the changes we'll do away with the original time column
df.drop("time", axis=1, inplace=True)

#Valencia_wind_deg column is another object
df['Valencia_wind_deg'] = df['Valencia_wind_deg'].str.extract(r'(\d+$)')
df["Valencia_wind_deg"] = pd.to_numeric(df["Valencia_wind_deg"])

#Seville_pressure is another object
df['Seville_pressure'] = df['Seville_pressure'].str.extract(r'(\d+$)')
df["Seville_pressure"] = pd.to_numeric(df["Seville_pressure"])

#Filling the Null cells with mode
df["Valencia_pressure"].fillna(train_data["Valencia_pressure"].mode()[0], inplace=True)


#The df should be split into test and train data now after the cleaning
ready_test_df = df[df["load_shortfall_3h"].isnull()]
ready_test_df = ready_test_df.drop("load_shortfall_3h", axis=1)
ready_train_df = df[df["load_shortfall_3h"].notnull()]

    """Private helper function to preprocess data for model prediction.
    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])
return ready_train_df

def load_model(assets/trained-models/Team_2_regression_model.pkl:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(assets/trained-models/Team_2_regression_model.pkl, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
