import numpy as np 
import pandas as pd
import awswrangler as wr
import ast
from pandas import json_normalize
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from msspackages import Pyspark_data_ingestion, get_features
from .utilities import S3Utilities
from .inputs import training_input, inference_input
from .evaluation import autoencoder_testing_pipeline, pca_testing_pipeline

    

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


def inference_data_builder(input_year, input_month,  input_day, input_hour, rec_type, input_setup):
    
    """
    inputs
    ------
            input_year : STRING | Int
            the year from which to read data, leave empty for all years

            input_month : STRING | Int
            the month from which to read data, leave empty for all months

            input_day : STRING | Int
            the day from which to read data, leave empty for all days

            input_hour: STRING | Int
            the hour from which to read data, leave empty for all hours
            
            rec_type: STRING
            uses schema for rec type when building pyspark df
            
            input_setup: STRING 
            kernel config
    
    outputs
    -------
            writes parquet to specific s3 path
    """

    if input_hour == -1:
        file_name = f'{rec_type}/{rec_type}_{input_year}_{input_month}_{input_day}'
    elif input_day == -1:
        file_name = f'{rec_type}/{rec_type}_{input_year}_{input_month}'
    else:
        file_name = f'{rec_type}/{rec_type}_{input_year}_{input_month}_{input_day}_{input_hour}'
    
    pyspark_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, setup = input_setup, filter_column_value = rec_type)
    err, pyspark_df = pyspark_data.read()
    
    if err == 'PASS':
        print(err)
        pyspark_df = pyspark_df.toPandas()
        wr.s3.to_parquet(pyspark_df, path=f"s3://dish-5g.core.pd.g.dp.eks.logs.e/inference_data/{file_name}.parquet")


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
    
    ##***Autoencoder***###

    #Inference for node autoencoder model
    inference_pipeline(node_inference_input(), node_autoencoder_input(), autoencoder_testing_pipeline)
    
    #Inference for pod autoencoder model
    inference_pipeline(pod_inference_input(), pod_autoencoder_input(), autoencoder_testing_pipeline)

    #Inference for container autoencoder model
    inference_pipeline(container_inference_input(), container_autoencoder_input(), autoencoder_testing_pipeline)
    
    ###***PCA***###
    
    #Inference for node pca model
    inference_pipeline(node_inference_input(), node_pca_input(), pca_testing_pipeline)
    
    #Inference for pod pca model
    inference_pipeline(pod_inference_input(), pod_pca_input(), pca_testing_pipeline)

    #Inference for container pca model
    inference_pipeline(container_inference_input(), container_pca_input(), pca_testing_pipeline)
    


    