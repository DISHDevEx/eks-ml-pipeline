import numpy as np
import pandas as pd
import random
from ..utilities import feature_processor, null_report
from msspackages import Pyspark_data_ingestion, get_features
from pyspark.sql.functions import col, count
from sklearn.preprocessing import StandardScaler

"""
Contributed by Vinayak Sharma and Praveen Mada
MSS Dish 5g - Pattern Detection

this feature engineering functions will help us run bach jobs that builds training data for Anomaly Detection models
"""


def node_autoencoder_ad_preprocessing(feature_group_name, feature_group_version, input_year, input_month, input_day, input_hour, input_setup = "default"):
    """
    inputs
    ------
            feature_group_name: STRING
            json name to get the required features
            
            feature_group_version: STRING
            json version to get the latest features 
            
            input_year : STRING | Int
            the year from which to read data, leave empty for all years

            input_month : STRING | Int
            the month from which to read data, leave empty for all months

            input_day : STRING | Int
            the day from which to read data, leave empty for all days

            input_hour: STRING | Int
            the hour from which to read data, leave empty for all hours
            
            input_setup: STRING 
            kernel config
    
    outputs
    -------
            features_df : processed features dataFrame
            processed_node_df: pre processed node dataframe
            
    """

    pyspark_node_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, setup = input_setup, filter_column_value ='Node')
    err, pyspark_node_df = pyspark_node_data.read()

    if err == 'PASS':

        #get features
        features_df = get_features(feature_group_name,feature_group_version)
        features = features_df["feature_name"].to_list()
        processed_features = feature_processor.cleanup(features)
    
        #filter inital node df based on request features
        node_df = pyspark_node_df.select("Timestamp", "InstanceId", *processed_features)
        node_df = node_df.withColumn("Timestamp",(col("Timestamp")/1000).cast("timestamp"))
        
        # Drop NA
        cleaned_node_df = node_df.na.drop(subset=processed_features)

        #Quality(timestamp filtered) nodes
        quality_filtered_node_df = cleaned_node_df.groupBy("InstanceId").agg(count("Timestamp").alias("timestamp_count"))
        # to get data that is closer to 1min apart
        quality_filtered_nodes = quality_filtered_node_df.filter(col("timestamp_count").between(45,75))
        
        #Processed Node DF                                                      
        processed_node_df = cleaned_node_df.filter(col("InstanceId").isin(quality_filtered_nodes["InstanceId"]))
        
        return features_df, processed_node_df

    else:
        empty_df = pd.DataFrame()
        return empty_df, empty_df
    

def node_autoencoder_ad_feature_engineering(input_data_type, input_split_ratio, input_node_features_df, input_node_processed_df):
    """
    inputs
    ------
            input_node_features_df: df
            processed node features df
            
            input_node_processed_df: df
            preprocessing and filtered node df 
    
    outputs
    -------
            node_tensor : np array for training the model
            final_node_fe_df: training data df for exposing it as data product
            
    """

    model_parameters = input_node_features_df["model_parameters"].iloc[0]
    features =  feature_processor.cleanup(input_node_features_df["feature_name"].to_list())

    time_steps = model_parameters["time_steps"]
    batch_size = model_parameters["batch_size"]

    if input_data_type == 'train':
        n_samples = batch_size * model_parameters["train_sample_multiplier"]
    elif input_data_type == 'test':
        n_samples = round((batch_size * model_parameters["train_sample_multiplier"]* input_split_ratio[1])/ input_split_ratio[0])

    node_tensor = np.zeros((n_samples,time_steps,len(features)))
    final_node_fe_df = pd.DataFrame()
    
    scaled_features = []
    for feature in features:
        scaled_features = scaled_features + ["scaled_"+feature]

    #To Pandas
    input_node_processed_df = input_node_processed_df.toPandas() 
    
    n = 0
    while n < n_samples:
        ##pick random df, and normalize
        random_instance_id = random.choice(input_node_processed_df["InstanceId"].unique())
        node_fe_df = input_node_processed_df.loc[(input_node_processed_df["InstanceId"] == random_instance_id)]
        node_fe_df = node_fe_df.sort_values(by='Timestamp').reset_index(drop=True)
        
        #scaler transformations
        scaler = StandardScaler()
        node_fe_df[scaled_features] = scaler.fit_transform(node_fe_df[features])
        
        node_fe_df_len = len(node_fe_df)
        
        #fix negative number bug 
        if node_fe_df_len-time_steps <= 0:
            print(f'Exception occurred: node_fe_df_len-time_steps = {node_fe_df_len-time_steps}')
            continue
        
        #tensor builder
        start = random.choice(range(node_fe_df_len-time_steps))
        node_tensor[n,:,:] = node_fe_df[start:start+time_steps][scaled_features]

        if final_node_fe_df.empty:
            final_node_fe_df = node_fe_df
        else:
            final_node_fe_df = final_node_fe_df.append(node_fe_df, ignore_index =True)

        print(f'Finished with sample #{n}')

        n +=1

    return final_node_fe_df, node_tensor


def node_autoencoder_train_test_split(input_df, split_weights):
    """
    inputs
    ------
            input_df: df
            processed/filtered input df from pre processing
            
    outputs
    -------
            node_train : train df
            node_test: test df
            
    """
    
    node_train, node_test = input_df.randomSplit(weights=split_weights, seed=200)

    return node_train, node_test
    