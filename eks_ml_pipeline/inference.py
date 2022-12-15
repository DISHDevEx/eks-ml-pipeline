import numpy as np 
import pandas as pd
import tensorflow as tf
from msspackages import Pyspark_data_ingestion, get_features
from utilities import write_tensor, read_tensor
from training_input import node_autoencoder_input, pod_autoencoder_input, container_autoencoder_input
from training_input import node_pca_input, pod_pca_input, container_pca_input
from sklearn.preprocessing import StandardScaler
from evaluation import autoencoder_testing_pipeline 

#Set random seed
#np.random.seed(10)

#raw_data_s3_path = 's3://dish-5g.core.pd.g.dp.eks.logs.e/inference_data/Node/Node_2022_9_11_12.parquet'
#raw_data_s3_path = 's3://dish-5g.core.pd.g.dp.eks.logs.e/inference_data/Container/Container_2022_8_20_9.parquet'
#raw_data_s3_path = 's3://dish-5g.core.pd.g.dp.eks.logs.e/inference_data/Pod/Pod_2022_7_10_20.parquet'

def read_raw_data(raw_data_s3_path):
    
    #Read raw data in parquet format from s3_path
    df = pd.read_parquet(raw_data_s3_path)
    
    print(f"reading raw data from: {raw_data_s3_path}")

    return df


def save_processed(df_raw: pd.DataFrame,
                   feature_group_name, feature_input_version,
                   data_bucketname, train_data_filename, test_data_filename,
                   save_model_local_path, model_bucketname,
                   model_name, model_version,
                   sampling_column = "InstanceId",
                   file_name = 'inference'):
                   
    #load data
    df = df_raw.copy()
    
    #Read features and parameters
    features_df = get_features(feature_group_name, feature_input_version)
    features = features_df["feature_name"].to_list()
    #remove spaces: that were put by mistake
    features = [feature.strip(' ') for feature in features]
    model_parameters = features_df["model_parameters"].iloc[0]
    time_steps = model_parameters["time_steps"]
    
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
    
    saved_file_name = ('_').join([file_name, sampling_column, random_id])
    
    write_tensor(tensor = inference_input_tensor, 
                 bucket_name = model_bucketname, 
                 model_name = model_name, 
                 version = model_version, 
                 flag = "data",
                 file_name = saved_file_name)
                                 
    return saved_file_name


if __name__ == "__main__":
    
    #Specify raw data s3 path
    raw_data_s3_path = 's3://dish-5g.core.pd.g.dp.eks.logs.e/inference_data/Node/Node_2022_9_11_12.parquet'
                                 
    #Read raw data
    df = read_raw_data(raw_data_s3_path)
    
    #load input parameters
    input_parameters = node_autoencoder_input()

    #Generate input tensor for a randomly selected sampling_column
    saved_file_name = save_processed(df, *input_parameters)
    
    #update save output path
    input_parameters[4] = saved_file_name
                                   
    predictions, residuals = autoencoder_testing_pipeline(*input_parameters[2:])
    