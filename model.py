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

def _preprocess_data(data):
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

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    feature_vector_df['time'] = pd.to_datetime(feature_vector_df['time'])
    feature_vector_df['Year'] = feature_vector_df['time'].dt.year
    feature_vector_df['Month'] = feature_vector_df['time'].dt.month
    feature_vector_df['Weekday'] = feature_vector_df['time'].dt.dayofweek
    feature_vector_df['Hour'] = feature_vector_df['time'].dt.hour
    feature_vector_df.drop("time", axis=1, inplace=True)
    feature_vector_df.drop("Unnamed: 0", axis=1, inplace=True)
    feature_vector_df['Valencia_wind_deg'] = feature_vector_df['Valencia_wind_deg'].str.extract(r'(\d+$)')
    feature_vector_df["Valencia_wind_deg"] = pd.to_numeric(feature_vector_df["Valencia_wind_deg"])
    feature_vector_df['Seville_pressure'] = feature_vector_df['Seville_pressure'].str.extract(r'(\d+$)')
    feature_vector_df["Seville_pressure"] = pd.to_numeric(feature_vector_df["Seville_pressure"])
    feature_vector_df["Valencia_pressure"].fillna(feature_vector_df["Valencia_pressure"].mode()[0], inplace=True)
    feature_vector=feature_vector_df.copy()
    
    feature_vector['wind_speed'] = feature_vector_df["Madrid_wind_speed"]+feature_vector_df["Valencia_wind_speed"] +feature_vector_df["Bilbao_wind_speed"]+feature_vector_df["Barcelona_wind_speed"]+feature_vector_df["Seville_wind_speed"]
    feature_vector["rain"] = feature_vector_df["Barcelona_rain_1h"]+ feature_vector_df["Barcelona_rain_3h"]+feature_vector_df["Seville_rain_1h"]+feature_vector_df["Seville_rain_3h"] + feature_vector_df["Bilbao_rain_1h"] + feature_vector_df["Madrid_rain_1h"]
    feature_vector["Clouds_all"]=feature_vector_df['Bilbao_clouds_all']+ feature_vector_df["Seville_clouds_all"]+ feature_vector_df["Madrid_clouds_all"]
    feature_vector['temp']=  feature_vector_df['Valencia_temp']+ feature_vector_df["Seville_temp"]+  feature_vector_df["Barcelona_temp"]+  feature_vector_df["Bilbao_temp"]+  feature_vector_df["Madrid_temp"]
    feature_vector["humidity"]=feature_vector_df["Seville_humidity"]+ feature_vector_df["Madrid_humidity"]+ feature_vector_df["Valencia_humidity"]

    
    
    predict_vector = feature_vector['wind_speed','rain', 'Clouds_all', 'temp', 'humidity', 'Year', 'Month', 'Weekday', 'Hour' ]
    # ------------------------------------------------------------------------

    return predict_vector

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
