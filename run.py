from feature_engineering import node_autoencoder_ad_preprocessing
import pandas as pd

data, features  = node_autoencoder_ad_preprocessing()
print(data.show())
pd.set_option('display.max_columns', None)
print(features)