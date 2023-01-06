import os
import numpy as np
from .utilities import S3Utilities
from .inputs import node_autoencoder_input, pod_autoencoder_input, container_autoencoder_input
from .inputs import node_pca_input, pod_pca_input, container_pca_input


"""
Contributed by Evgeniya Dontsova and Vinayak Sharma
MSS Dish 5g - Pattern Detection

this model testing functions will be used to test Anomaly Detection models and save the predictions
"""

def model_evaluation_pipeline(encode_decode_model,
                              feature_group_name, feature_input_version, 
                              data_bucketname, train_data_filename, test_data_filename,
                              save_model_local_path, model_bucketname, model_filename,
                              upload_zip, upload_onnx, upload_npy,
                              clean_local_folder = True):
        
   
    """
    Generalized model evaluation pipeline

    inputs
    ------
            encode_decode_model: Class
            initialized model class object that has 
            fit(), predict(), save_model(), load_model(), 
            and clean_model() methods

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
            
            model_filename: str
            model filename where training model will be saved
           
            upload_zip: bool
            flag to save model in zip format
            
            upload_onnx: bool
            flag to save model in onnx format

            upload_npy: bool
            flag to save model in npy format
            
            clean_local_folder: bool
            flag to delete or keep locally saved model directory or files
    
    outputs
    -------
            None
            
    """

    
    ###Initialize s3 utilities class
    s3_utils = S3Utilities(bucket_name=data_bucketname, 
                           model_name=feature_group_name, 
                           version=feature_input_version)


    ###Load training data: read from s3 bucket
    testing_tensor = s3_utils.read_tensor(folder = "data", 
                                          type_ = "tensors", 
                                          file_name = test_data_filename)
    
    
    #Additional data cleaning: converting everything into np.float32
    testing_tensor = np.asarray(testing_tensor).astype(np.float32)

        
    ###Load trained model: read from s3 bucket
    if upload_zip:
        s3_utils.download_zip(local_path = save_model_local_path + '.zip',
                              folder = "models",
                              type_ = "zipped_models",
                              file_name = model_filename + '.zip')
        
        s3_utils.unzip(path_to_zip = save_model_local_path + '.zip',
                       extract_location = save_model_local_path)


    if upload_npy:
        load_tensor = s3_utils.read_tensor(folder = "models",
                                           type_ = "npy_models", 
                                           file_name = model_filename + ".npy")
        np.save(save_model_local_path, load_tensor)
            

    ###Load trained model from local path
    encode_decode_model.load_model(save_model_local_path)

    ###Use trained model to predict for testing tensor
    results = encode_decode_model.predict(testing_tensor)
    
    ###Save predictions
    for i, result in enumerate(results):
        print(f'predictions_{test_data_filename.split(".")[-2]}_part_{i}.npy')
        s3_utils.write_tensor(tensor = result, 
                              folder = "models", 
                              type_ = "predictions", 
                              file_name = f'predictions_{test_data_filename.split(".")[-2]}_part_{i}.npy')
        
        
    #Delete locally saved model
    if clean_local_folder:
        
        encode_decode_model.clean_model(save_model_local_path)
        
        if upload_zip:
            
            path = save_model_local_path + '.zip'
            os.remove(path)
            print(f"\n***Locally saved {path} was succesfully deleted.***\n")
     

    return None


if __name__ == "__main__":
    
    ##***Autoencoder***###

    #Test node autoencoder model and save on s3
    model_evaluation_pipeline(*node_autoencoder_input())
    
    #Test pod autoencoder model and save on s3
    model_evaluation_pipeline(*pod_autoencoder_input())

    #Test container autoencoder model and save on s3
    model_evaluation_pipeline(*container_autoencoder_input())
    
    ##***PCA***###
    
    #Test node pca model and save on s3
    model_evaluation_pipeline(*node_pca_input())
    
    #Test pod pca model and save on s3
    model_evaluation_pipeline(*pod_pca_input())

    #Test container pca model and save on s3
    model_evaluation_pipeline(*container_pca_input())