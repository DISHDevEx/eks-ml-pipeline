import importlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
import sys
import statsmodels.api as sm
import os
import keras
from keras import layers
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from pca_ad_dish_5g import pca_ad_dish_5g

from sklearn.model_selection import train_test_split

import _pickle as cPickle



if __name__ == "__main__":
    
  
    
    ##global variable
    timesteps = 12
    time_steps = 12
    batch_size = 6
    n_samples = batch_size*100
    features = ['node_cpu_utilization','node_memory_utilization','node_network_total_bytes']

    pca_model = pca_ad_dish_5g(num_of_features=3, number_of_temporal_slices=3, timesteps_per_slice=4, n_modes_to_delete=1)
    pca_model.load_in_vs('vs.npy')
    
    
    
    ##read in data 
    df_full = pd.read_parquet('/root/healthy_clusters_node_month.parquet')
    columns_to_keep = ['Timestamp','InstanceId','node_cpu_utilization','node_memory_utilization','node_network_total_bytes']
    df_full = df_full.drop(df_full.columns.difference(columns_to_keep),1, inplace=False)
    df_full['Timestamp'] = pd.to_datetime(df_full['Timestamp'], unit='ms')

    ## in this dataset .007% of all rows have a null. So we will drop them quickly since they do not effect the dataset as a whole. For larger processing, we should apply more dataquality filters
    """
    (3) iterate through all nodeID Dataframes:

        drop nulls['node_cpu_utilization','node_memory_utilization','node_network_total_bytes'] in nodeID Dataframe


        if nodeID Dataframe has: 
            a) Number of timestamps >= 60
            c) Max time delta <= 75 seconds
            d) Min time delta >= 45 seconds
        store NodeID dataframe

    """
    df_full = df_full.dropna()
    
    
    ###CREATE TEST SET (Holdout set)
    test_df = df_full[df_full.InstanceId == 'i-0b36e8825c482f762']



    ## drop the instanceId
    test_df = test_df.drop("InstanceId",1, inplace=False)


    ##set timestamp as the index
    test_df = test_df.set_index('Timestamp')

    ##normalize test_df 
    scaler = StandardScaler()
    test_df[features] = scaler.fit_transform(test_df[features])


    ##ensure the data is sorted!!
    test_df = test_df.sort_index()
    
    
    ##actual inferencing
    y_hat_test = []
    res_test = []

    for i in range(0,12,timesteps):
        if(i + timesteps < len(test_df)):
            sample_topredict_on = test_df.iloc[i:i+timesteps]
            x_test = np.array(sample_topredict_on.to_numpy())
            x_test = x_test.reshape(1,-1,3)
            y_hat,residuals = pca_model.test(x_test)
            y_hat_reshape = y_hat.reshape(1,12,3) 
            res_reshape = residuals.reshape(1,12,3) 
            y_hat_test.append(y_hat_reshape)
            res_test.append(res_reshape)

