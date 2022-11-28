from feature_engineering import node_autoencoder_ad_preprocessing, node_autoencoder_ad_feature_engineering
from utilities import write_to_s3
import pandas as pd

features_data, processed_data = node_autoencoder_ad_preprocessing("node_autoencoder_ad","11-21-2022","2022","10","10","10")

training_data = node_autoencoder_ad_feature_engineering(features_data, processed_data)

print(training_data.show(truncate=False))

# pd.set_option('display.max_columns', None)  
# print(features.head())

#write_to_s3(training_data)