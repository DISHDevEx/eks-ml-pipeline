import numpy as np 
import pandas as pd
import ast
from pandas import json_normalize
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from msspackages import get_features
from utilities import write_tensor, read_tensor
from training_input import node_autoencoder_input, pod_autoencoder_input, container_autoencoder_input
from training_input import node_pca_input, pod_pca_input, container_pca_input
from inference_input import node_inference_input, pod_inference_input, container_inference_input
from evaluation import autoencoder_testing_pipeline, pca_testing_pipeline

"""
Contributed by Evgeniya Dontsova
MSS Dish 5g - Pattern Detection

this inference functions will be used for Anomaly Detection models
"""


def only_dict(d):
    """
    inputs
    ------
            d: dict

    outputs
    -------
            Convert json string representation of dictionary to a python dict

    """
    return ast.literal_eval(d)


def read_raw_data(raw_data_s3_path):
    """
    inputs
    ------
            raw_data_s3_path: str

    outputs
    -------
            df: pd.DataFrame

    """

    #Read raw data in parquet format from s3_path
    df = pd.read_parquet(raw_data_s3_path)
    
    print(f"\nreading raw data from: {raw_data_s3_path}\n")

    return df

def build_processed_data(inference_input_parameters,
                         training_input_parameters):
    
    """
    inputs
    ------
            inference_input_parameters: list
            list of parameters required for inference

            training_input_parameters: list
            list of training input parameter specific to rec type and model
                
    
    outputs
    -------
            training_input_parameters: list
            updated list of training input parameter specific to rec type and model
            
    """

    
    #inference input 
    raw_data_s3_path, sampling_column, file_prefix = inference_input_parameters
    
    #training input 
    feature_group_name, feature_input_version, \
    data_bucketname, train_data_filename, test_data_filename, \
    save_model_local_path, model_bucketname, \
    model_name, model_version = training_input_parameters
                   
    #load data
    df = read_raw_data(raw_data_s3_path)
        
    #Read features and parameters
    features_df = get_features(feature_group_name, feature_input_version)
    features = features_df["feature_name"].to_list()
    #remove spaces: that were put by mistake
    features = [feature.strip(' ') for feature in features]
    model_parameters = features_df["model_parameters"].iloc[0]
    time_steps = model_parameters["time_steps"]
    
    #drop null values
    df = df.dropna(subset = features, axis = 0)
    
    #check sampling column: if needs unpacking
    if type(sampling_column) == tuple:
        #unpack
        df_unpack = json_normalize(df[sampling_column[0]].apply(only_dict)\
                                  .tolist()).add_prefix(sampling_column[0] + '.')
        df = df.join(df_unpack)
        sampling_column = '.'.join(sampling_column)
       
    sampling_column_unique = df[sampling_column].unique()
    
    if len(sampling_column_unique) >=1:
        #select unique sampling_column (e.g. InstanceId for Node or pod_id for Pod
        random_id = np.random.choice(df[sampling_column].unique())
        print(f'\n*** Select data with unique {sampling_column} = {random_id} ***\n')
        df = df.loc[(df[sampling_column] == random_id)]
        #sort by time
        df = df.sort_values(by='Timestamp').reset_index(drop=True)

        #select last time slice of data
        start = df.shape[0] - time_steps
        df = df.loc[start:start+time_steps, features]

        print("\n***** Inference input data shape*****")
        print(df.shape)
        print("\n*** Inference data tensor ***")
        print(df)
        print("\n***************************************\n")

        #scaler transformations
        scaler = StandardScaler()
        scaled_features = ["scaled_" + feature for feature in features]
        df[scaled_features] = scaler.fit_transform(df[features])
        inference_input_tensor = np.expand_dims(df[scaled_features], axis = 0)

        print("\n***** Inference input tensor shape*****")
        print(inference_input_tensor.shape)
        print("\n*** Inference input tensor ***")
        print(inference_input_tensor)
        print("\n***************************************\n")

        saved_file_name = ('_').join([file_prefix, sampling_column, random_id])

        write_tensor(tensor = inference_input_tensor, 
                     bucket_name = model_bucketname, 
                     model_name = model_name, 
                     version = model_version, 
                     flag = "data",
                     file_name = saved_file_name)

        #update training input parameters with new test_data_filename
        training_input_parameters[4] = saved_file_name
        
    else:
        print(f"Exception occured: no unique values for {sampling_column} column.")

    return training_input_parameters

def inference_pipeline(inference_input_parameters,
                       training_input_parameters,
                       prediction_pipeline):
    
    """
    inputs
    ------
            inference_input_parameters: list
            list of parameters required for inference

            training_input_parameters: list
            list of training input parameter specific to rec type and model
            
            prediction_pipeline: function
            available values are autoencoder_testing_pipeline() or pca_testing_pipeline()

    outputs
    -------
            predictions: np.array
            model predictions
            
            residuals: np.array
            model residuals
            
    """

    
    training_input_parameters = build_processed_data(inference_input_parameters, training_input_parameters)
    
    predictions, residuals = prediction_pipeline(*training_input_parameters)
    
    return predictions, residuals

if __name__ == "__main__":
    
    ###***Autoencoder***###

#     #Inference for node autoencoder model
#     inference_pipeline(node_inference_input(), node_autoencoder_input(), autoencoder_testing_pipeline)
    
    # #Inference for pod autoencoder model
    # inference_pipeline(pod_inference_input(), pod_autoencoder_input(), autoencoder_testing_pipeline)

    #Inference for container autoencoder model
    inference_pipeline(container_inference_input(), container_autoencoder_input(), autoencoder_testing_pipeline)
    
    ###***PCA***###
    
    #Inference for node pca model
    inference_pipeline(node_inference_input(), node_pca_input(), pca_testing_pipeline)
    
    #Inference for pod pca model
    inference_pipeline(pod_inference_input(), pod_pca_input(), pca_testing_pipeline)

    #Inference for container pca model
    inference_pipeline(container_inference_input(), container_pca_input(), pca_testing_pipeline)
    


    