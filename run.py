from feature_engineering import node_autoencoder_ad_preprocessing
import pandas as pd

data = node_autoencoder_ad_preprocessing("node_autoencoder_ad_features","11-21-2022","2022","10","10","10")
print(data.show(truncate=False))
