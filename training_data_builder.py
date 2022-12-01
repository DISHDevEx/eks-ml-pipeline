

from feature_engineering import node_autoencoder_ad_preprocessing, node_autoencoder_ad_feature_engineering, node_autoencoder_train_test_split
from feature_engineering import container_autoencoder_ad_preprocessing, container_autoencoder_ad_feature_engineering, container_autoencoder_train_test_split
from feature_engineering import pod_autoencoder_ad_preprocessing, pod_autoencoder_ad_feature_engineering, pod_autoencoder_train_test_split


from utilities import s3_operations
import pandas as pd


def node_training_data_builder():
    
    #pre processing
    pod_features_data, pod_processed_data = pod_autoencoder_ad_preprocessing("pod_autoencoder_ad","11-21-2022","2022","10","10","10")
    #pd.set_option('display.max_columns', None)  
    # print(pod_features_data.head())
    # print(pod_processed_data.show(truncate=False))
    
    #test, train split
    pod_train_data, pod_test_data = pod_autoencoder_train_test_split(pod_processed_data)

    #Train data feature engineering
    pod_training_data = pod_autoencoder_ad_feature_engineering(pod_features_data, pod_train_data)
    #s3_operations.write_to_s3(pod_training_data, s3_path)
    print(pod_training_data.show(truncate=False))
 
    #Test data feature engineering
    pod_testing_data = pod_autoencoder_ad_feature_engineering(pod_features_data, pod_test_data)
    #s3_operations.write_to_s3(pod_testing_data, s3_path)
    print(pod_testing_data.show(truncate=False))

    
def pod_training_data_builder():
    
    #pre processing
    pod_features_data, pod_processed_data = pod_autoencoder_ad_preprocessing("pod_autoencoder_ad","11-30-2022","2022","10","10","10")
    #pd.set_option('display.max_columns', None)  
    print(pod_features_data.head())
    print(pod_processed_data.show(truncate=False))
    
    #test, train split
    pod_train_data, pod_test_data = pod_autoencoder_train_test_split(pod_processed_data)

    #Train data feature engineering
    pod_training_data = pod_autoencoder_ad_feature_engineering(pod_features_data, pod_train_data)
    #s3_operations.write_to_s3(pod_training_data, s3_path)
    print(pod_training_data.show(truncate=False))
 
    #Test data feature engineering
    pod_testing_data = pod_autoencoder_ad_feature_engineering(pod_features_data, pod_test_data)
    #s3_operations.write_to_s3(pod_testing_data, s3_path)
    print(pod_testing_data.show(truncate=False))
    
    
def container_training_data_builder():
    
    #pre processing
    container_features_data, container_processed_data = container_autoencoder_ad_preprocessing("container_autoencoder_ad","11-21-2022","2022","10","10","10")
    # pd.set_option('display.max_columns', None)  
    # print(container_features_data.head())
    # print(container_processed_data.show(truncate=False))
    
    #test, train split
    container_train_data, container_test_data = container_autoencoder_train_test_split(container_processed_data)

    #Train data feature engineering
    container_training_data = container_autoencoder_ad_feature_engineering(container_features_data, container_train_data)
    #s3_operations.write_to_s3(pod_training_data, s3_path)
    print(container_training_data.show(truncate=False))
 
    #Test data feature engineering
    container_testing_data = container_autoencoder_ad_feature_engineering(container_features_data, container_test_data)
    #s3_operations.write_to_s3(pod_testing_data, s3_path)
    print(container_testing_data.show(truncate=False))
    
    
if __name__ == "__main__":
    #pod_training_data_builder()
    pod_training_data_builder()