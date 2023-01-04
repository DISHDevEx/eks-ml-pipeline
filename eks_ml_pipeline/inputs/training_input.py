from ..models import autoencoder_model_dish_5g, pca_model_dish_5g
from msspackages import get_features


"""
Contributed by Evgeniya Dontsova and Vinayak Sharma
MSS Dish 5g - Pattern Detection

these are the parameters to our pipeline


"""

def node_autoencoder_input_all():
    
    """
    outputs
    -------
            list of parameters for node rec type
            required by autoencoder model 
            training pipeline
            
    """

    #*****************************************************#
    #********** data and model input parameters **********#
    #*****************************************************#

    feature_group_name = "node_autoencoder_ad"
    feature_input_version = "v0.0.2"  
    data_bucketname = "dish-5g.core.pd.g.dp.eks.logs.e"
    train_data_filename = "training_2022_9_29.npy" #add info about actual path
    test_data_filename = "testing_2022_9_29.npy" #add info about actual path
    
    save_model_local_path = "../../node_autoencoder"
    model_bucketname = 'dish-5g.core.pd.g.dp.eks.logs.e'
    model_name = 'node_autoencoder_ad'
    model_version = "test" #'v0.0.2' #add info about actual path for model
    
    #model s3 bucket upload format
    upload_zip = True 
    upload_onnx = True
    upload_npy = False
    
    #********************************************#
    #********** initialize model class **********#
    #********************************************#
    
    features_df = get_features(feature_group_name, feature_input_version)
    model_parameters = features_df["model_parameters"].iloc[0]
    
    #Initialize autoencoder model class with specific parameters
    encode_decode_model = autoencoder_model_dish_5g(time_steps=model_parameters["time_steps"], 
                                                    batch_size=model_parameters["batch_size"], epochs=1)

    
    return [encode_decode_model,
            feature_input_version, data_bucketname, 
            train_data_filename, test_data_filename,
            save_model_local_path, 
            model_bucketname, model_name, model_version,
            upload_zip, upload_onnx, upload_npy]


def node_pca_input_all():
    """
    outputs
    -------
            list of parameters for node rec type
            required by pca model 
            training pipeline
            
    """
    
    #*****************************************************#
    #********** data and model input parameters **********#
    #*****************************************************#

    ##generate pipeline input params for pca
    feature_group_name = "node_pca_ad"
    feature_input_version = "v0.0.1"  #need to add print statement if version does not exists
    data_bucketname = 'dish-5g.core.pd.g.dp.eks.logs.e'
    train_data_filename = "training_2022_9_29.npy"
    test_data_filename = "testing_2022_9_29.npy"

    save_model_local_path = "../../pod_pca.npy"
    model_bucketname = "dish-5g.core.pd.g.dp.eks.logs.e"
    model_name = "node_pca_ad"
    model_version = "test"
    
    #model s3 bucket upload format
    upload_zip = False 
    upload_onnx = False
    upload_npy = True
    
    #********************************************#
    #********** initialize model class **********#
    #********************************************#
    
    features_df = get_features(feature_group_name, feature_input_version)
    features = features_df["feature_name"].to_list()
    model_parameters = features_df["model_parameters"].iloc[0]
    
    #Initialize pca model 
    encode_decode_model = pca_model_dish_5g(num_of_features = len(features), 
                                            timesteps_per_slice = model_parameters["time_steps"] )


    return [encode_decode_model,
            feature_input_version, data_bucketname, 
            train_data_filename, test_data_filename,
            save_model_local_path, 
            model_bucketname, model_name, model_version,
            upload_zip, upload_onnx, upload_npy]


def node_autoencoder_input():
    
    """
    outputs
    -------
            list of parameters for node rec type
            required by autoencoder model 
            training pipeline
            
    """

    
    feature_group_name = "node_autoencoder_ad"
    feature_input_version = "v0.0.2"  
    data_bucketname = "dish-5g.core.pd.g.dp.eks.logs.e"
    train_data_filename = "training_2022_9_29.npy" #add info about actual path
    test_data_filename = "testing_2022_9_29.npy" #add info about actual path
    
    ##always save these 1 level out
    save_model_local_path = "../../node_autoencoder"
    model_bucketname = 'dish-5g.core.pd.g.dp.eks.logs.e'
    model_name = 'node_autoencoder_ad'
    model_version = "test" #'v0.0.2' 
    #add info about actual path for model
    
    return [feature_group_name, feature_input_version,
            data_bucketname, train_data_filename, test_data_filename,
            save_model_local_path, model_bucketname,
            model_name, model_version]

def node_pca_input():
    """
    outputs
    -------
            list of parameters for node rec type
            required by pca model 
            training pipeline
            
    """
    ##generate pipeline input params for pca
    feature_group_name = "node_pca_ad"
    feature_input_version = "v0.0.1"  
    data_bucketname = 'mss-shared'
    train_data_filename = "pcaDummyData"
    test_data_filename = "pcaDummyData"

    save_model_local_path = "../../node_pca.npy"
    model_bucketname = "mss-shared"
    model_name = "node_pca_ad"
    model_version = "v0.0.1"

    return [feature_group_name, feature_input_version,
            data_bucketname, train_data_filename,test_data_filename,
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
    feature_input_version = "v0.0.2"  
    data_bucketname = 'mss-shared'
    train_data_filename = 'aeDummyData'
    test_data_filename = 'aeDummyData'
    
    save_model_local_path = "../../pod_autoencoder"
    model_bucketname = 'mss-shared'
    model_name = 'pod_autoencoder_ad'
    model_version = 'v0.0.2'
    
    return [feature_group_name, feature_input_version,
            data_bucketname, train_data_filename, test_data_filename,
            save_model_local_path, model_bucketname,
            model_name, model_version]

def pod_pca_input():
    """
    outputs
    -------
            list of parameters for pod rec type
            required by pca model 
            training pipeline
            
    """
    ##generate pipeline input params for pca
    feature_group_name = "pod_pca_ad"
    feature_input_version = "v0.0.1"  
    data_bucketname = 'mss-shared'
    train_data_filename = 'pcaDummyData'
    test_data_filename = 'pcaDummyData'

    save_model_local_path = "../../pod_pca.npy"
    model_bucketname = 'mss-shared'
    model_name = 'pod_pca_ad'
    model_version = 'v0.0.1'

    return [feature_group_name, feature_input_version,
            data_bucketname, train_data_filename,test_data_filename,
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
    feature_input_version = "v0.0.2"  
    data_bucketname = 'mss-shared'
    train_data_filename = 'aeDummyData'
    test_data_filename = 'aeDummyData'
    
    save_model_local_path = "../../container_autoencoder"
    model_bucketname = 'mss-shared'
    model_name = 'container_autoencoder_ad'
    model_version = 'v0.0.2'
    
    return [feature_group_name, feature_input_version,
            data_bucketname, train_data_filename, test_data_filename,
            save_model_local_path, model_bucketname,
            model_name, model_version]

def container_pca_input():
    """
    outputs
    -------
            list of parameters for container rec type
            required by pca model 
            training pipeline
            
    """
    ##generate pipeline input params for pca
    feature_group_name = "container_pca_ad"
    feature_input_version = "v0.0.1"  
    data_bucketname = 'mss-shared'
    train_data_filename = 'pcaDummyData'
    test_data_filename = 'pcaDummyData'

    save_model_local_path = "../../container_pca.npy"
    model_bucketname = 'mss-shared'
    model_name = 'container_pca_ad'
    model_version = 'v0.0.1'

    return [feature_group_name, feature_input_version,
            data_bucketname, train_data_filename,test_data_filename,
            save_model_local_path, model_bucketname,
            model_name, model_version]
