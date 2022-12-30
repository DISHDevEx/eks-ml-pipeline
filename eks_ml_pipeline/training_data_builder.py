from .feature_engineering import node_autoencoder_ad_preprocessing, node_autoencoder_ad_feature_engineering, node_autoencoder_train_test_split 
from .feature_engineering import pod_autoencoder_ad_preprocessing, pod_autoencoder_ad_feature_engineering, pod_autoencoder_train_test_split
from .feature_engineering import container_autoencoder_ad_preprocessing, container_autoencoder_ad_feature_engineering, container_autoencoder_train_test_split
from .feature_engineering import node_hmm_ad_preprocessing, node_hmm_ad_feature_engineering, node_hmm_train_test_split
import pandas as pd


def node_training_data_builder():
    """
    Output
    ------
        node_training_data:df
        training df to be stored in s3

        node_testing_data:df
        testing df to be stores in s3

        node_training_tensor: np array
        tensor stored in s3 for model training

        node_testing_tensor: np array
        tensor stored in s3 for model testing

    """
    
    #pre processing
    node_features_data, node_processed_data = node_autoencoder_ad_preprocessing("node_autoencoder_ad","v0.0.1","2022","10","10","10","128gb")
    
    #test, train split
    node_train_split = node_features_data["model_parameters"].iloc[0]["split_ratio"]
    node_test_split =  round(1 - node_train_split,2)
    node_train_data, node_test_data = node_autoencoder_train_test_split(node_processed_data, [node_train_split,node_test_split])

    #Train data feature engineering
    node_training_data, node_training_tensor = node_autoencoder_ad_feature_engineering('train', [node_train_split,node_test_split], node_features_data, node_train_data)
 
    #Test data feature engineering
    node_testing_data, node_testing_tensor = node_autoencoder_ad_feature_engineering('test', [node_train_split,node_test_split],  node_features_data, node_test_data)
    
    return node_training_data, node_training_tensor, node_testing_data, node_testing_tensor

    
def pod_training_data_builder():
    """
    Output
    ------
        pod_training_data:df
        training df to be stored in s3

        pod_testing_data:df
        testing df to be stores in s3

        pod_training_tensor: np array
        tensor stored in s3 for model training

        pod_testing_tensor: np array
        tensor stored in s3 for model testing

    """
    
    #pre processing
    pod_features_data, pod_processed_data = pod_autoencoder_ad_preprocessing("pod_autoencoder_ad","v0.0.1","2022","10","10","10","128gb")
    
    #test, train split
    pod_train_split = pod_features_data["model_parameters"].iloc[0]["split_ratio"]
    pod_test_split =  round(1 - pod_train_split,2)
    pod_train_data, pod_test_data = pod_autoencoder_train_test_split(pod_processed_data, [pod_train_split,pod_test_split])

    #Train data feature engineering
    pod_training_data, pod_training_tensor = pod_autoencoder_ad_feature_engineering('train', [pod_train_split,pod_test_split], pod_features_data, pod_train_data)
 
    #Test data feature engineering
    pod_testing_data, pod_testing_tensor = pod_autoencoder_ad_feature_engineering('test', [pod_train_split,pod_test_split], pod_features_data, pod_test_data)
    
    return pod_training_data, pod_training_tensor, pod_testing_data, pod_testing_tensor


def container_training_data_builder():
    """
    Output
    -------
    
        container_training_data:df
        training df to be stored in s3

        container_testing_data:df
        testing df to be stores in s3

        container_training_tensor: np array
        tensor stored in s3 for model training

        container_testing_tensor: np array
        tensor stored in s3 for model testing
    
    """
    
    #pre processing
    container_features_data, container_processed_data = container_autoencoder_ad_preprocessing("container_autoencoder_ad","v0.0.1","2022","10","10","10","128gb")
    
    #test, train split
    container_train_split = container_features_data["model_parameters"].iloc[0]["split_ratio"]
    container_test_split =  round(1 - container_train_split,2)
    container_train_data, container_test_data = container_autoencoder_train_test_split(container_processed_data, [container_train_split,container_test_split])

    #Train data feature engineering
    container_training_data, container_training_tensor = container_autoencoder_ad_feature_engineering('train', [container_train_split,container_test_split], container_features_data, container_train_data)
 
    #Test data feature engineering
    container_testing_data, container_testing_tensor = container_autoencoder_ad_feature_engineering('test', [container_train_split,container_test_split], container_features_data, container_test_data)
    
    return container_training_data, container_training_tensor, container_testing_data, container_testing_tensor
    

def node_hmm_training_data_builder():
    """
    Output
    ------
        node_hmm_training_data:df
        training df to be stored in s3

        node_hmm_testing_data:df
        testing df to be stores in s3

    """
    
    #pre processing
    node_hmm_features_data, node_hmm_processed_data = node_hmm_ad_preprocessing("node_hmm_ad","v0.0.1","2022","10","10","10","128gb")
    
    #test, train split
    node_train_split = node_features_data["model_parameters"].iloc[0]["split_ratio"]
    node_test_split =  round(1 - node_train_split,2)
    node_hmm_train_data, node_hmm_test_data = node_hmm_train_test_split(node_hmm_processed_data, [node_train_split,node_test_split])

    #Train data feature engineering
    node_training_data = node_hmm_ad_feature_engineering('train', [node_train_split,node_test_split], node_hmm_features_data, node_hmm_train_data)
 
    #Test data feature engineering
    node_testing_data = node_hmm_ad_feature_engineering('test', [node_train_split,node_test_split],  node_hmm_features_data, node_hmm_test_data)
    
    return node_training_data, node_testing_data
    
if __name__ == "__main__":
    #node_training_data_builder()
    pod_training_data_builder()