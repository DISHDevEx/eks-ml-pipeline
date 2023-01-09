import numpy as np
import tf2onnx
from .utilities import S3Utilities
from .inputs import training_input


"""
Contributed by Evgeniya Dontsova and Vinayak Sharma
MSS Dish 5g - Pattern Detection

this model training functions will be used to train and save Anomaly Detection models
"""

def model_training_pipeline(encode_decode_model,
                            feature_group_name, feature_input_version, 
                            data_bucketname, train_data_filename, test_data_filename,
                            save_model_local_path, model_bucketname, model_filename,
                            upload_zip, upload_onnx, upload_npy,
                            clean_local_folder):
        
    """
    Generalized model training pipeline

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
    training_tensor = s3_utils.read_tensor(folder = "data", 
                                           type_ = "tensors", 
                                           file_name = train_data_filename)
    
    #Additional data cleaning: converting everything into np.float32
    training_tensor = np.asarray(training_tensor).astype(np.float32)
            
    ###Train model
    model = encode_decode_model.fit(training_tensor)
    
    ###Save model
    encode_decode_model.save_model(save_model_local_path)
    

    ####Save model object to s3 bucket
    #save zipped model object to s3 bucket   
    if upload_zip:
        
        s3_utils.zip_and_upload(local_path = save_model_local_path, 
                                folder = "models", 
                                type_ = "zipped_models", 
                                file_name = model_filename + ".zip")
        
    #save onnx model object to s3 bucket   
    if upload_onnx:
        
        save_model_local_path_onnx = save_model_local_path + '/' + model_filename + ".onnx"
        #Save model locally in .onnx format 
        model_proto, external_tensor_storage = tf2onnx.convert.from_keras(encode_decode_model.nn,
                                                                          output_path = save_model_local_path_onnx)
        s3_utils.upload_file(local_path = save_model_local_path_onnx, 
                             bucket_name = model_bucketname, 
                             key = '/'.join([feature_group_name, feature_input_version,
                                             "models", "onnx_models", model_filename + ".onnx"]))
        
    #save npy model object to s3 bucket          
    if upload_npy:
        
        s3_utils.write_tensor(tensor = model, 
                              folder = "models", 
                              type_ = "npy_models", 
                              file_name = model_filename + ".npy")
        
    #Delete locally saved model
    if clean_local_folder:
        
        encode_decode_model.clean_model(save_model_local_path)
    
    
    return None
