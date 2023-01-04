import numpy as np
import boto3
import tf2onnx
from io import BytesIO
from msspackages import get_features
from .utilities import S3Utilities
from .models import autoencoder_model_dish_5g, pca_model_dish_5g
from .inputs import training_input


"""
Contributed by Evgeniya Dontsova and Vinayak Sharma
MSS Dish 5g - Pattern Detection

this model training functions will be used to train and save Anomaly Detection models
"""

def model_training_pipeline(encode_decode_model,
                            feature_group_name, feature_input_version, 
                            data_bucketname, 
                            train_data_filename, test_data_filename,
                            save_model_local_path, 
                            model_bucketname, model_name, model_version,
                            upload_zip, upload_onnx, upload_npy):
    
    """
    Generalized model training pipeline
    """
    
    ###Initialize s3 utilities class
    s3_utils = S3Utilities(bucket_name=data_bucketname, 
                           model_name="node_autoencoder_ad", 
                           version="v0.0.2")

        
    ###Load training data: read from s3 bucket
    training_tensor = s3_utils.read_tensor(folder = "data", 
                                           type_ = "tensors", 
                                           file_name = train_data_filename)
    
    # print(training_tensor)
    # print(training_tensor.shape)
        
    ###Train model
    encode_decode_model.fit(training_tensor)
    
    ###Save model
    encode_decode_model.save_model(save_model_local_path)
    
    #Define model file name
    model_file_name = '_'.join([model_name, "model", model_version, train_data_filename.split(".")[-2]])


    ####Save model object to s3 bucket
    #save zipped model object to s3 bucket   
    if upload_zip:
        
        s3_utils.zip_and_upload(local_path = save_model_local_path, 
                                folder = "models", 
                                type_ = "zipped_models", 
                                file_name = model_file_name + ".zip")
        
    #save onnx model object to s3 bucket   
    if upload_onnx:
        
        save_model_local_path_onnx = save_model_local_path + '/' + model_file_name + ".onnx"
        #Save model locally in .onnx format 
        model_proto, external_tensor_storage = tf2onnx.convert.from_keras(encode_decode_model.nn,
                                                                          output_path = save_model_local_path_onnx)
        s3_utils.upload_file(local_path = save_model_local_path_onnx, 
                             bucket_name = model_bucketname, 
                             key = '/'.join([model_name, feature_input_version, "models", model_file_name + ".onnx"]))
        
    #save npy model object to s3 bucket          
    if upload_npy:
        
        s3_utils.write_tensor(save_model_local_path, 
                              folder = "models", 
                              type_ = "npy_models", 
                              file_name = model_file_name + ".zip")
    
    
    return None



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
            autoencoder: autoencoder_model_dish_5g class object
            trained keras model
            
    """

    
    features_df = get_features(feature_group_name, feature_input_version)
    model_parameters = features_df["model_parameters"].iloc[0]
    
    #Initialize autoencoder model
    autoencoder = autoencoder_model_dish_5g(time_steps=model_parameters["time_steps"], 
                                            batch_size=model_parameters["batch_size"], epochs=1)
    
    #Train model
    autoencoder.fit(training_tensor)
    
    #Save model
    autoencoder.save_nn(save_model_local_path)
        
    return autoencoder


def autoencoder_training_pipeline(feature_group_name, feature_input_version,
                                  data_bucketname, train_data_filename, test_data_filename,
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
            
            train_data_filename: str
            filename where training data is saved 
                        
            test_data_filename: str
            filename where testing data is saved 

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
            trained autoencoder model
            
    """
    
    ###Initialize s3 utilities class
    s3_utils = S3Utilities(bucket_name=data_bucketname, 
                           model_name=model_name, 
                           version=feature_input_version)

        
    ###Load training data: read from s3 bucket
    training_tensor = s3_utils.read_tensor(folder = "data", 
                                           type_ = "tensors", 
                                           file_name = train_data_filename)
    

    ####Train autoencoder model
    autoencoder = autoencoder_training(training_tensor, 
                                       feature_group_name, 
                                       feature_input_version, 
                                       save_model_local_path)
    
    #Define model file name
    model_file_name = '_'.join([model_name, "model", model_version, train_data_filename.split(".")[-2]])


    ####Save model object to s3 bucket

    s3_utils.zip_and_upload(local_path = save_model_local_path, 
                            folder = "models", 
                            type_ = "zipped_models", 
                            file_name = model_file_name + ".zip")
            
    
    ###Save model locally in .onnx format 
    save_model_local_path_onnx = save_model_local_path + '/' + model_file_name + ".onnx"
    model_proto, external_tensor_storage = tf2onnx.convert.from_keras(autoencoder.nn,
                                                                      output_path = save_model_local_path_onnx)
    ####Save onnx model object to s3 bucket   
    s3_utils.upload_file(local_path = save_model_local_path_onnx, 
                         bucket_name = model_bucketname, 
                         key = '/'.join([model_name, feature_input_version, "models", model_file_name + ".onnx"]))
     
    return autoencoder
    

def pca_training(training_tensor, 
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
            pca: pca_dish_5g class object
            
    """

    
    features_df = get_features(feature_group_name, feature_input_version)
    features = features_df["feature_name"].to_list()
    model_parameters = features_df["model_parameters"].iloc[0]
    
    #Initialize pca model 
    pca = pca_model_dish_5g(num_of_features = len(features), timesteps_per_slice = model_parameters["time_steps"] )
    
    #Train model
    pca.fit(training_tensor)
    
    #Save model
    pca.save_vs(save_model_local_path)
        
    return pca


def pca_training_pipeline(feature_group_name, feature_input_version,
                          data_bucketname, train_data_filename,test_data_filename,
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
            trained pca model
            
    """    
        

    
    ###Load training data: read from s3 bucket
    training_tensor = read_tensor(bucket_name = data_bucketname,
                                  model_name = model_name,
                                  version = model_version,
                                  file_name =  train_data_filename)

    ####Train autoencoder model
    pca = pca_training(training_tensor, 
                                       feature_group_name, 
                                       feature_input_version, 
                                       save_model_local_path)


    pca_vs = np.load(save_model_local_path)
    
    ####Save model object to s3 bucket
    write_tensor(tensor = pca_vs,
                    bucket_name = model_bucketname,
                    model_name = model_name,
                    version = model_version,
                    flag = "model",
                    file_name = '_'.join([model_name, model_version, train_data_filename]))

    
if __name__ == "__main__":
    
    ###***Autoencoder***###
    
#     #Train node autoencoder model and save on s3
#     autoencoder_training_pipeline(*node_autoencoder_input())
    
#     #Train pod autoencoder model and save on s3
#     autoencoder_training_pipeline(*pod_autoencoder_input())

#     #Train container autoencoder model and save on s3
#     autoencoder_training_pipeline(*container_autoencoder_input())
    
#     ###***PCA***###
    
#     #Train node pca model and save on s3
#     pca_training_pipeline(*node_pca_input())
    
#     #Train pod pca model and save on s3
#     pca_training_pipeline(*pod_pca_input())

#     #Train container pca model and save on s3
#     pca_training_pipeline(*container_pca_input())


    model_training_pipeline(*node_autoencoder_input_all())