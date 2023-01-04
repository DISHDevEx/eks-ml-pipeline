"""
Contributed by Evgeniya Dontsova and Vinayak Sharma
MSS Dish 5g - Pattern Detection

these are the parameters to our pipeline


"""

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
    data_bucketname = "dish-5g.core.pd.g.dp.eks.logs.e"  #'mss-shared'
    train_data_filename = "training_2022_9_29.npy"
    test_data_filename = "testing_2022_9_29.npy"
    
    ##always save these 1 level out
    save_model_local_path = "../../node_autoencoder"
    model_bucketname = 'mss-shared'
    model_name = 'node_autoencoder_ad'
    model_version = 'v0.0.2'
    
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
