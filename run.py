from feature_engineering import node_autoencoder_ad_preprocessing, node_autoencoder_ad_feature_engineering, node_autoencoder_train_test_split, container_autoencoder_ad_preprocessing, container_autoencoder_ad_feature_engineering
from utilities import s3_operations
import pandas as pd


def node_training_data_builder():
    
    #pre processing
    node_features_data, node_processed_data = node_autoencoder_ad_preprocessing("node_autoencoder_ad","11-21-2022","2022","10","10","10")
    #pd.set_option('display.max_columns', None)  
    # print(node_features_data.head())
    # print(node_processed_data.show(truncate=False))
    
    #test, train split
    node_train_data, node_test_data = node_autoencoder_train_test_split(node_processed_data)

    #Train data feature engineering
    node_training_data = node_autoencoder_ad_feature_engineering(node_features_data, node_train_data)
    #s3_operations.write_to_s3(node_training_data, s3_path)
    print(node_training_data.show(truncate=False))
 
    #Test data feature engineering
    node_testing_data = node_autoencoder_ad_feature_engineering(node_features_data, node_test_data)
    #s3_operations.write_to_s3(node_testing_data, s3_path)
    print(node_testing_data.show(truncate=False))

    
    
    
def container_training_data_builder():
    
    #pre processing
    container_features_data, container_processed_data = container_autoencoder_ad_preprocessing("container_autoencoder_ad","11-21-2022","2022","10","10","10")
    # pd.set_option('display.max_columns', None)  
    # print(container_features_data.head())
    # print(container_processed_data.show(truncate=False))

    #feature engineering
    container_training_data = container_autoencoder_ad_feature_engineering(container_features_data, container_processed_data)
    print(container_training_data.show(truncate=False))

    #s3_operations.write_to_s3(training_data, s3_path)
    
    
if __name__ == "__main__":
    node_training_data_builder()