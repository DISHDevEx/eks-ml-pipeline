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





if __name__ == "__main__":
    
    
    
    ##load in model
    with open(r"eks_Trained_Autoencoder.pickle", "rb") as input_file:
        model = cPickle.load(input_file)
    
    ##global variable
    timesteps = 12
    time_steps = 12
    batch_size = 6
    n_samples = batch_size*100
    features = ['node_cpu_utilization','node_memory_utilization','node_network_total_bytes']



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
    predictions_f1 = []
    predictions_f2 = []
    predictions_f3 = []

    anomaly_scoresf1 = []
    anomaly_scoresf2 = []
    anomaly_scoresf3 = []

    errors_f1 = []
    errors_f2 = []
    errors_f3 = []


    for i in range(0,len(test_df),timesteps):
        if(i + timesteps < len(test_df)):
            sample_topredict_on = test_df.iloc[i:i+timesteps]
            x_test = np.array(sample_topredict_on.to_numpy())
            x_test = x_test.reshape(1,-1,3)
            preds,errs,anom_scores = model.test(x_test)

            predictions_f1.append(np.array(preds[:,:,0]))
            predictions_f2.append(preds[:,:,1])
            predictions_f3.append(preds[:,:,2])

            anomaly_scoresf1.append(np.array(anom_scores[:timesteps,:]))
            anomaly_scoresf2.append(anom_scores[timesteps:timesteps+timesteps,:])
            anomaly_scoresf3.append(anom_scores[timesteps+timesteps:timesteps+timesteps+timesteps,:])


            errors_f1.append(np.array(errs[:,:,0]))
            errors_f2.append(errs[:,:,1])
            errors_f3.append(errs[:,:,2])

