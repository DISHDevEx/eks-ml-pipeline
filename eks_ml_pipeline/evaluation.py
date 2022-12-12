import numpy as np
import tensorflow as tf
import pyarrow.parquet as pq
from eks_ml_pipeline import autoencoder_model_dish_5g, write_tensor, read_tensor
from eks_ml_pipeline import node_autoencoder_input, pod_autoencoder_input, container_autoencoder_input

"""
Contributed by Evgeniya Dontsova
MSS Dish 5g - Pattern Detection

this model testing functions will be used to test Anomaly Detection models and save the predictions
"""


def autoencoder_testing_pipeline(data_bucketname, train_data_filename, test_data_filename,
                                 save_model_local_path, model_bucketname,
                                 model_name, model_version):
    
    """
    inputs
    ------
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
            autoencoder: keras model object
            trained keras model
            
    """

    
    ###Load testing data: read from s3 bucket
    testing_tensor = read_tensor(data_bucketname,
                                 test_data_filename)
    
    #Load trained model
    autoencoder = tf.keras.models.load_model(save_model_local_path)
    
    #Make predictions
    test_predictions = autoencoder.predict(testing_tensor)
    
    #Calculate residuals for testing data == anomaly score
    test_residuals = np.abs(test_predictions - testing_tensor)
    
    #Write test_predictions tensor
    write_tensor(test_predictions, model_bucketname, 
                 model_name, model_version, 
                 filename = 'test_predictions.npy')
    
    #Write test_residuals tensor
    write_tensor(test_residuals, model_bucketname, 
                 model_name, model_version, 
                 filename = 'test_residuals.npy')
    
    return test_predictions, test_residuals
    

if __name__ == "__main__":
    
    #Train node autoencoder model and save on s3
    autoencoder_testing_pipeline(*node_autoencoder_input()[2:])
    
    #Train pod autoencoder model and save on s3
    autoencoder_testing_pipeline(*pod_autoencoder_input()[2:])

    #Train container autoencoder model and save on s3
    autoencoder_testing_pipeline(*container_autoencoder_input()[2:])