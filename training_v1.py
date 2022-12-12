import numpy as np
import boto3
from io import BytesIO
from msspackages import get_features
from eks_ml_pipeline import read_tensor,write_tensor,uploadDirectory, autoencoder_model_dish_5g, pca_model_dish_5g
from training_input_v1 import node_autoencoder_input,node_pca_input,pod_autoencoder_input,pod_pca_input,container_autoencoder_input,container_pca_input

"""
Contributed by Evgeniya Dontsova and Vinayak Sharma
MSS Dish 5g - Pattern Detection

these model training functions will be used to train and save Anomaly Detection models
"""
####################################################
###********TRAINING PIPELINE******** ###
####################################################

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
                                  data_bucketname, train_data_filename,test_data_filename,
                                  save_model_local_path, model_bucketname,
                                  model_name, model_version):
    """
    inputs
    ------
            model_name:str
            name of the model being trained in this pipeline
            
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
                                  train_data_filename)

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
            trained keras model
            
    """

    
    features_df = get_features(feature_group_name, feature_input_version)
    model_parameters = features_df["model_parameters"].iloc[0]
    
    #Initialize autoencoder model
    pca = pca_model_dish_5g(num_of_features =3, number_of_temporal_slices = 5, timesteps_per_slice = 4)
    
    #Train model
    pca.train(training_tensor)
    
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
            model_name:str
            name of the model being trained in this pipeline
            
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
                      train_data_filename)

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
                    filename = model_name + model_version)

        

####################################################
###******** MAIN RUNNER ******** ###
####################################################
    
if __name__ == "__main__":
    
#     #Train node autoencoder model and save on s3
#     autoencoder_training_pipeline(*node_autoencoder_input())
#     #Train node autoencoder model and save on s3
#     pca_training_pipeline(*node_pca_input())
    
    
    
#     #Train pod autoencoder model and save on s3
#     autoencoder_training_pipeline(*pod_autoencoder_input(())
#     #Train pod autoencoder model and save on s3
#     pca_training_pipeline(*pod_pca_input(())
                                  

#     #Train container autoencoder model and save on s3
#     autoencoder_training_pipeline(*container_autoencoder_input(())
#     #Train container autoencoder model and save on s3
    pca_training_pipeline(*container_pca_input())