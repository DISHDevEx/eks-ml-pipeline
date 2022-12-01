from feature_engineering import node_autoencoder_ad_preprocessing, node_autoencoder_ad_feature_engineering, node_autoencoder_train_test_split, container_autoencoder_ad_preprocessing, container_autoencoder_ad_feature_engineering, container_autoencoder_train_test_split
import pandas as pd


def node_training_data_builder():
    
    #pre processing
    node_features_data, node_processed_data = node_autoencoder_ad_preprocessing("node_autoencoder_ad","11-21-2022","2022","10","10","10","128gb")
    #pd.set_option('display.max_columns', None)  
    # print(node_features_data.head())
    # print(node_processed_data.show(truncate=False))
    
    #test, train split
    node_train_data, node_test_data = node_autoencoder_train_test_split(node_processed_data)

    #Train data feature engineering
    node_training_data = node_autoencoder_ad_feature_engineering(node_features_data, node_train_data)
    print(node_training_data.show(truncate=False))
 
    #Test data feature engineering
    node_testing_data = node_autoencoder_ad_feature_engineering(node_features_data, node_test_data)
    print(node_testing_data.show(truncate=False))
    
    return node_training_data, node_testing_data

    
    
    
def container_training_data_builder():
    
    #pre processing
    container_features_data, container_processed_data = container_autoencoder_ad_preprocessing("container_autoencoder_ad","11-21-2022","2022","10","10","10","128gb")
    # pd.set_option('display.max_columns', None)  
    # print(container_features_data.head())
    # print(container_processed_data.show(truncate=False))
    
    #test, train split
    container_train_data, container_test_data = container_autoencoder_train_test_split(container_processed_data)

    #Train data feature engineering
    container_training_data = container_autoencoder_ad_feature_engineering(container_features_data, container_train_data)
    print(container_training_data.show(truncate=False))
 
    #Test data feature engineering
    container_testing_data = container_autoencoder_ad_feature_engineering(container_features_data, container_test_data)
    print(container_testing_data.show(truncate=False))
    
    return container_training_data, container_testing_data
    
    
if __name__ == "__main__":
    node_training_data_builder()