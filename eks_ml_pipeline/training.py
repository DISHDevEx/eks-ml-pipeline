import numpy as np
import boto3
from io import BytesIO
from msspackages import get_features
from eks_ml_pipeline import read_tensor, uploadDirectory, autoencoder_model_dish_5g

"""
Contributed by Evgeniya Dontsova
MSS Dish 5g - Pattern Detection

this model training functions will be used to train and save Anomaly Detection models
"""


def autoencoder_training(training_tensor, 
                         feature_group_name, 
                         feature_input_version, 
                         save_model_local_path):
    
    """
    inputs
    ------
            training_tensor: np.array
            tensor requires as model input for training

            feature_group_name: str
            json name to get the required features
            
            feature_input_version: str
            json version to get the latest features 
            
            save_model_local_path: str
            local path to save trained model 
            (by default it is one level above git repo)
    
    
    outputs
    -------
            autoencoder: keras model object
            trained keras model
            
    """

    
    features_df = get_features(feature_group_name, feature_input_version)
    model_parameters = features_df["model_parameters"].iloc[0]
    
    #Initialize autoencoder model
    autoencoder = autoencoder_model_dish_5g(time_steps=model_parameters["time_steps"], 
                                            batch_size=model_parameters["batch_size"])
    
    #Train model
    autoencoder.fit(training_tensor)
    
    #Save model
    autoencoder.save_nn(save_model_local_path)
        
    return autoencoder


def autoencoder_training_pipeline(feature_group_name, feature_input_version,
                                  data_bucketname, data_filename,
                                  save_model_local_path, model_bucketname,
                                  model_name, model_version):
    
    """
    inputs
    ------
            feature_group_name: str
            json name to get the required features
            
            feature_input_version: str
            json version to get the latest features 
            
            data_bucketname: str
            s3 bucket name where training data is saved
            
            data_filename: str
            filename where training data is saved 

            save_model_local_path: str
            local path to save trained model 
            (by default it is one level above git repo)
            
            model_bucketname: str
            s3 bucket name where trained model will be saved
            
            model_name: str
            filename where training model will be saved

            model_version: str
            training model version

    
    outputs
    -------
            autoencoder: keras model object
            trained keras model
            
    """

        
    ###Load training data: read from s3 bucket
    training_tensor = read_tensor(data_bucketname,
                                  data_filename)
    
    ####Train autoencoder model
    autoencoder = autoencoder_training(training_tensor, 
                                       feature_group_name, 
                                       feature_input_version, 
                                       save_model_local_path)


    ####Save model object to s3 bucket
    uploadDirectory(local_path = save_model_local_path,
                    bucketname = model_bucketname,
                    model_name = model_name,
                    version = model_version)
    

####################################################
###******** AUTOENCODER MODEL PARAMETERS******** ###
####################################################

def node_autoencoder_input():
    
    """
    outputs
    -------
            list of parameters for node rec type
            required by autoencoder model 
            training pipeline
            
    """

    
    feature_group_name = "node_autoencoder_ad"
    feature_input_version = "v0.0.1"  
    data_bucketname = 'mss-shared'
    data_filename = 'x_train_36k_sample.npy'
    
    save_model_local_path = "../node_autoencoder"
    model_bucketname = 'emr-serverless-output-pd'
    model_name = 'node_autoencoder_test'
    model_version = 'v0.0.1'
    
    return [feature_group_name, feature_input_version,
            data_bucketname, data_filename,
            save_model_local_path, model_bucketname,
            model_name, model_version]

def pod_autoencoder_input():
    
    """
    outputs
    -------
            list of parameters for pod rec type
            required by autoencoder model 
            training pipeline
            
    """

    
    feature_group_name = "pod_autoencoder_ad"
    feature_input_version = "v0.0.1"  
    data_bucketname = 'mss-shared'
    data_filename = 'x_train_36k_sample.npy'
    
    save_model_local_path = "../pod_autoencoder"
    model_bucketname = 'emr-serverless-output-pd'
    model_name = 'pod_autoencoder_test'
    model_version = 'v0.0.1'
    
    return [feature_group_name, feature_input_version,
            data_bucketname, data_filename,
            save_model_local_path, model_bucketname,
            model_name, model_version]

def container_autoencoder_input():
    
    """
    outputs
    -------
            list of parameters for container rec type
            required by autoencoder model 
            training pipeline
            
    """

    
    feature_group_name = "container_autoencoder_ad"
    feature_input_version = "v0.0.1"  
    data_bucketname = 'mss-shared'
    data_filename = 'x_train_36k_sample.npy'
    
    save_model_local_path = "../container_autoencoder"
    model_bucketname = 'emr-serverless-output-pd'
    model_name = 'container_autoencoder_test'
    model_version = 'v0.0.1'
    
    return [feature_group_name, feature_input_version,
            data_bucketname, data_filename,
            save_model_local_path, model_bucketname,
            model_name, model_version]

    
if __name__ == "__main__":
    
    #Train node autoencoder model and save on s3
    autoencoder_training_pipeline(*node_autoencoder_input())
    
    #Train pod autoencoder model and save on s3
    autoencoder_training_pipeline(*pod_autoencoder_input())

    #Train container autoencoder model and save on s3
    autoencoder_training_pipeline(*container_autoencoder_input())