from feature_engineering import node_autoencoder_ad_preprocessing, node_autoencoder_ad_feature_engineering, write_to_s3
import pandas as pd

processed_data = node_autoencoder_ad_preprocessing("node_autoencoder_ad_features","11-21-2022","2022","10","10","10")

training_data = node_autoencoder_ad_feature_engineering(processed_data)

print(training_data.show(truncate=False))

write_to_s3(training_data)