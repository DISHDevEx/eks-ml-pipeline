from feature_engineering import node_autoencoder_ad_preprocessing, node_autoencoder_ad_feature_engineering
from feature_engineering import container_autoencoder_ad_preprocessing, container_autoencoder_ad_feature_engineering
from feature_engineering import pod_autoencoder_ad_preprocessing, pod_autoencoder_ad_feature_engineering

from utilities import s3_operations
import pandas as pd


def node_training_data_builder():
    
    #pre processing
    node_features_data, node_processed_data = node_autoencoder_ad_preprocessing("node_autoencoder_ad","11-21-2022","2022","10","10","10")
    pd.set_option('display.max_columns', None)  
    print(node_features_data.head())
    print(node_processed_data.show(truncate=False))

    # #feature engineering
    # node_training_data = node_autoencoder_ad_feature_engineering(node_features_data, node_processed_data)
    # print(node_training_data.show(truncate=False))

    #s3_operations.write_to_s3(training_data)
    
def pod_training_data_builder():
    
    #pre processing
    pod_features_data, pod_processed_data = pod_autoencoder_ad_preprocessing("pod_autoencoder_ad","11-30-2022","2022","10","10","10")
    pd.set_option('display.max_columns', None)  
    print(pod_features_data.head())
    print(pod_processed_data.show(truncate=False))

#     #feature engineering
#     pod_training_data = pod_autoencoder_ad_feature_engineering(pod_features_data, pod_processed_data)
#     print(pod_training_data.show(truncate=False))

    #s3_operations.write_to_s3(training_data)
     
    
def container_training_data_builder():
    
    #pre processing
    container_features_data, container_processed_data = container_autoencoder_ad_preprocessing("container_autoencoder_ad","11-21-2022","2022","10","10","10")
    pd.set_option('display.max_columns', None)  
    print(container_features_data.head())
    print(container_processed_data.show(truncate=False))

    #feature engineering
    container_training_data = container_autoencoder_ad_feature_engineering(container_features_data, container_processed_data)
    print(container_training_data.show(truncate=False))

    #s3_operations.write_to_s3(training_data)
    
    
if __name__ == "__main__":
    
#     print("\n\n\n node_training_data_builder() \n\n\n")
#     node_training_data_builder()
    
    print("\n\n\n pod_training_data_builder() \n\n\n")
    pod_training_data_builder()