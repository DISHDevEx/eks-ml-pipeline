from feature_engineering import node_autoencoder_ad_preprocessing, node_autoencoder_ad_feature_engineering, node_autoencoder_train_test_split, container_autoencoder_ad_preprocessing, container_autoencoder_ad_feature_engineering, container_autoencoder_train_test_split
import pandas as pd


def node_training_data_builder():
    
    #pre processing
    node_features_data, node_processed_data = node_autoencoder_ad_preprocessing("node_autoencoder_ad","11-21-2022","2022","10","10","10","128gb")
    
    #test, train split
    node_train_data, node_test_data = node_autoencoder_train_test_split(node_processed_data)

    #Train data feature engineering
    node_training_data, node_training_tensor = node_autoencoder_ad_feature_engineering(node_features_data, node_train_data)
 
    #Test data feature engineering
    node_testing_data, node_testing_tensor = node_autoencoder_ad_feature_engineering(node_features_data, node_test_data)
    
    return node_training_data, node_training_tensor, node_testing_data, node_testing_tensor

    
    
    
def container_training_data_builder():
    
    #pre processing
    container_features_data, container_processed_data = container_autoencoder_ad_preprocessing("container_autoencoder_ad","11-21-2022","2022","10","10","10","128gb")
    
    #test, train split
    container_train_data, container_test_data = container_autoencoder_train_test_split(container_processed_data)

    #Train data feature engineering
    container_training_data, container_training_tensor = container_autoencoder_ad_feature_engineering(container_features_data, container_train_data)
 
    #Test data feature engineering
    container_testing_data, container_testing_tensor = container_autoencoder_ad_feature_engineering(container_features_data, container_test_data)
    
    return container_training_data, container_training_tensor, container_testing_data, container_testing_tensor
    
    
if __name__ == "__main__":
    node_training_data_builder()