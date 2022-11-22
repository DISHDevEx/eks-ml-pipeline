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
from autoencoder_model_dish_5g import Autoencoder_Model_Dish_5g

from sklearn.model_selection import train_test_split

import _pickle as cPickle


if __name__ == "__main__":
    ##global variable
    timesteps = 12
    time_steps = 12
    batch_size = 6
    n_samples = batch_size*100
    features = ['node_cpu_utilization','node_memory_utilization','node_network_total_bytes']


    ##read in data 
    training_df_full = pd.read_parquet('/root/healthy_clusters_node_month.parquet')
    columns_to_keep = ['Timestamp','InstanceId','node_cpu_utilization','node_memory_utilization','node_network_total_bytes']
    training_df = training_df_full.drop(training_df_full.columns.difference(columns_to_keep),1, inplace=False)
    training_df['Timestamp'] = pd.to_datetime(training_df['Timestamp'], unit='ms')

    ## in this dataset .007% of all rows have a null. So we will drop them quickly since they do not effect the dataset as a whole. For larger processing, we should apply more dataquality filters
    """
     more dataquality filters:
     
    -->iterate through all nodeID Dataframes:

        drop nulls['node_cpu_utilization','node_memory_utilization','node_network_total_bytes'] in nodeID Dataframe


        if nodeID Dataframe has: 
            a) Number of timestamps >= 60
            c) Max time delta <= 75 seconds
            d) Min time delta >= 45 seconds
        store NodeID dataframe

    """
    training_df = training_df.dropna()
    test_df = training_df.copy()
    training_df = training_df[training_df.InstanceId != 'i-0b36e8825c482f762']


    ## for normalization
    # scaler = StandardScaler()

    scaler = StandardScaler()

    instance_dfs =[]
    for instance in training_df['InstanceId'].unique():
        instance_dfs.append(training_df[training_df.InstanceId == instance].sort_values(by='Timestamp')\
                            .reset_index(drop=True))

    import random 

    x_train = np.zeros((n_samples,time_steps,len(features)))
    for b in range(n_samples):

        ##pick random df, and normalize
        df = random.choice(instance_dfs)
        df = df.drop(columns = ['InstanceId'])
        df = df.set_index('Timestamp')
        df = df.sort_index()
        df[features] = scaler.fit_transform(df[features])



        sample = np.zeros((n_samples,len(features)))
        ##make sure length of df is atleast 40
        first_time = random.choice(range(len(df)-time_steps))
        df.head()
        sample = df[features].iloc[first_time:first_time+time_steps]
        x_train[b] = sample

    x_train.shape


    model = Autoencoder_Model_Dish_5g(batch_size = batch_size, time_steps = time_steps)
    val_err = model.train(x_train)
    
    with open(r"eks_Trained_Autoencoder.pickle", "wb") as output_file:
        cPickle.dump(model, output_file)
    