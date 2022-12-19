import numpy as np
import tensorflow as tf
from msspackages import get_features
from utilities import write_tensor, read_tensor
from models import pca_model_dish_5g
from training_input import node_autoencoder_input, pod_autoencoder_input, container_autoencoder_input
from training_input import node_pca_input, pod_pca_input, container_pca_input


"""
Contributed by Evgeniya Dontsova and Vinayak Sharma
MSS Dish 5g - Pattern Detection

this model testing functions will be used to test Anomaly Detection models and save the predictions
"""


def autoencoder_testing_pipeline(feature_group_name, feature_input_version,
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
            test_predictions: np.array
            model predictions
            
            test_residuals: np.array
            model residuals
            
    """

    
    ###Load testing data: read from s3 bucket
    testing_tensor = read_tensor(bucket_name = data_bucketname,
                                  model_name = model_name,
                                  version = model_version,
                                  file_name =  test_data_filename)
    
    #Load trained model
    autoencoder = tf.keras.models.load_model(save_model_local_path)
    
    #Make predictions
    test_predictions = autoencoder.predict(testing_tensor)
    
    #Calculate residuals for testing data == anomaly score
    test_residuals = np.abs(test_predictions - testing_tensor)
    
    print("\n*** Autoencoder residulas/anomaly score ***\n")
    print(test_residuals[0,:,:])
    print("\n*******************************************\n")
    
    file_name = ('_').join(['input_tensor', test_data_filename])

    #Write test_predictions tensor
    write_tensor(test_predictions,
                 model_bucketname,
                 model_name,
                 model_version,
                 "data", "predictions_" + file_name)
    
    #Write test_residuals tensor
    write_tensor(test_residuals,
                 model_bucketname,
                 model_name,
                 model_version,
                 "data","residuals_" + file_name)
    
    return test_predictions, test_residuals

def pca_testing_pipeline(feature_group_name, feature_input_version,
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
            test_predictions: np.array
            model predictions
            
            test_residuals: np.array
            model residuals
            
    """
    
    features_df = get_features(feature_group_name, feature_input_version)
    features = features_df["feature_name"].to_list()
    model_parameters = features_df["model_parameters"].iloc[0]

    
    ###Load testing data: read from s3 bucket
    testing_tensor = read_tensor(bucket_name = data_bucketname,
                                  model_name = model_name,
                                  version = model_version,
                                  file_name =  test_data_filename)
    
    #Load trained model 
    pca = pca_model_dish_5g(num_of_features = len(features), timesteps_per_slice = model_parameters["time_steps"])
    pca.load_in_vs(save_model_local_path)
    
    #Make predictions
    test_predictions,test_residuals = pca.test(testing_tensor)
    
    file_name = ('_').join(['input_tensor', test_data_filename])

    
    #Write test_predictions tensor
    write_tensor(test_predictions,
                 model_bucketname,
                 model_name,
                 model_version,
                 "data", "predictions_" + file_name)
    
    #Write test_residuals tensor
    write_tensor(test_residuals,
                 model_bucketname,
                 model_name,
                 model_version,
                 "data", "residuals_" + file_name)
    
    return test_predictions, test_residuals    

if __name__ == "__main__":
    
    ##***Autoencoder***###

    #Test node autoencoder model and save on s3
    autoencoder_testing_pipeline(*node_autoencoder_input())
    
    #Test pod autoencoder model and save on s3
    autoencoder_testing_pipeline(*pod_autoencoder_input())

    #Test container autoencoder model and save on s3
    autoencoder_testing_pipeline(*container_autoencoder_input())
    
    ##***PCA***###
    
    #Test node pca model and save on s3
    pca_testing_pipeline(*node_pca_input())
    
    #Test pod pca model and save on s3
    pca_testing_pipeline(*pod_pca_input())

    #Test container pca model and save on s3
    pca_testing_pipeline(*container_pca_input())