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
    train_data_filename = 'x_train_36k_sample.npy'
    test_data_filename = 'x_train_36k_sample.npy'
    
    save_model_local_path = "../node_autoencoder"
    model_bucketname = 'emr-serverless-output-pd'
    model_name = 'node_autoencoder_test'
    model_version = 'v0.0.1'
    
    return [feature_group_name, feature_input_version,
            data_bucketname, train_data_filename, test_data_filename,
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
    train_data_filename = 'x_train_36k_sample.npy'
    test_data_filename = 'x_train_36k_sample.npy'
    
    save_model_local_path = "../pod_autoencoder"
    model_bucketname = 'emr-serverless-output-pd'
    model_name = 'pod_autoencoder_test'
    model_version = 'v0.0.1'
    
    return [feature_group_name, feature_input_version,
            data_bucketname, train_data_filename, test_data_filename,
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
    train_data_filename = 'x_train_36k_sample.npy'
    test_data_filename = 'x_train_36k_sample.npy'
    
    save_model_local_path = "../container_autoencoder"
    model_bucketname = 'emr-serverless-output-pd'
    model_name = 'container_autoencoder_test'
    model_version = 'v0.0.1'
    
    return [feature_group_name, feature_input_version,
            data_bucketname, train_data_filename, test_data_filename,
            save_model_local_path, model_bucketname,
            model_name, model_version]