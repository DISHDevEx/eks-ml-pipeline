import numpy as np
import pandas as pd
import random
from ..utilities import feature_processor, null_report
from msspackages import Pyspark_data_ingestion, get_features
from pyspark.sql.functions import col, count, row_number, get_json_object
from sklearn.preprocessing import StandardScaler

"""
Contributed by Madhu Bandi, Evgeniya Dontsova and Praveen Mada
MSS Dish 5g - Pattern Detection

this feature engineering functions will help us run bach jobs that builds training data for Anomaly Detection models
"""

def pod_autoencoder_ad_preprocessing(feature_group_name, feature_group_version, input_year, input_month, input_day, input_hour, input_setup = "default"):
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
            final_pod_df: pre processed node dataframe
            
    """

    pod_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, setup = input_setup, filter_column_value ='Pod')
    err, pod_df = pod_data.read()
    pod_df = pod_df.select(*pod_df.columns,
                           get_json_object(col("kubernetes"),"$.pod_id").alias("pod_id"),
                           col("pod_status"))

 
    if err == 'PASS':
        #get features
        features_df = get_features(feature_group_name,feature_group_version)
        features = features_df["feature_name"].to_list()
        processed_features = feature_processor.cleanup(features)
        
        model_parameters = features_df["model_parameters"].iloc[0]
        time_steps = model_parameters["time_steps"]
    
    
        #filter inital pod df based on request features
        pod_df = pod_df.select("Timestamp", "pod_id", "pod_status", *processed_features)
        pod_df = pod_df.withColumn("Timestamp",(col("Timestamp")/1000).cast("timestamp"))
        cleaned_pod_df = pod_df.na.drop(subset=processed_features)
        
        #Quality(timestamp filtered) pods
        cleaned_pod_df = cleaned_pod_df.filter(col("pod_status") == "Running")
        quality_filtered_pod_df = cleaned_pod_df.groupBy("pod_id").agg(count("Timestamp").alias("timestamp_count"))
        quality_filtered_pods = quality_filtered_pod_df.filter(col("timestamp_count") >= 2*time_steps)

        #Processed pod DF                                                      
        final_pod_df = cleaned_pod_df.filter(col("pod_id").isin(quality_filtered_pods["pod_id"]))
        final_pod_df = final_pod_df.sort("Timestamp")
                
        #Drop duplicates on Pod_ID and Timestamp and keep first
        final_pod_df = final_pod_df.dropDuplicates(['pod_id', 'Timestamp'])
        
        #Drop rows with nans 
        final_pod_df = final_pod_df.na.drop("all")
           
        
        return features_df, final_pod_df
    else:
        empty_df = pd.DataFrame()
        return empty_df
        


def pod_autoencoder_ad_feature_engineering(input_data_type, input_split_ratio, input_pod_features_df, input_pod_processed_df):
    """
    inputs
    ------
            input_pod_features_df: df
            processed node features df
            
            input_pod_processed_df: df
            preprocessing and filtered node df 
    
    outputs
    -------
            pod_tensor : np array for training the model
            final_pod_fe_df: training data df for exposing it as data product
            
    """

    model_parameters = input_pod_features_df["model_parameters"].iloc[0]
    features =  feature_processor.cleanup(input_pod_features_df["feature_name"].to_list())
    
    time_steps = model_parameters["time_steps"]
    batch_size = model_parameters["batch_size"]
    if input_data_type == 'train':
        n_samples = batch_size * model_parameters["train_sample_multiplier"]
    elif input_data_type == 'test':
         n_samples = round((batch_size * model_parameters["train_sample_multiplier"]* input_split_ratio[1])/ input_split_ratio[0])

    pod_tensor = np.zeros((n_samples,time_steps,len(features)))
    final_pod_fe_df = pd.DataFrame()
    
    scaled_features = []
    for feature in features:
        scaled_features = scaled_features + ["scaled_"+feature]

    #To Pandas
    input_pod_processed_df = input_pod_processed_df.toPandas()
    
    n = 0
    while n < n_samples:
        ##pick random df, and normalize
        random_pod_id = random.choice(input_pod_processed_df["pod_id"].unique())
        pod_fe_df = input_pod_processed_df.loc[(input_pod_processed_df["pod_id"] == random_pod_id)]
        pod_fe_df = pod_fe_df.sort_values(by='Timestamp').reset_index(drop=True)
        pod_fe_df_len = len(pod_fe_df)
        
        #fix negative number bug 
        if pod_fe_df_len-time_steps <= 0:
            print(f'Exception occurred: pod_fe_df_len-time_steps = {pod_fe_df_len-time_steps}')
            continue

        #tensor builder
        start = random.choice(range(pod_fe_df_len-time_steps))
        pod_fe_df = pod_fe_df[start:start+time_steps]
        
        #scaler transformations
        scaler = StandardScaler()
        pod_fe_df[scaled_features] = scaler.fit_transform(pod_fe_df[features])
        pod_tensor[n,:,:] = pod_fe_df[scaled_features]

        if final_pod_fe_df.empty:
            final_pod_fe_df = pod_fe_df
        else:
            final_pod_fe_df = final_pod_fe_df.append(pod_fe_df, ignore_index =True)

        print(f'Finished with sample #{n}')

        n +=1

    return final_pod_fe_df, pod_tensor

    

def pod_autoencoder_train_test_split(input_df, split_weights):
    
    """
    inputs
    ------
            input_df: df
            processed/filtered input df from pre processing
            
    outputs
    -------
            pod_train : train df
            pod_test: test df
            
    """
    
    
    pod_train, pod_test = input_df.randomSplit(weights= split_weights, seed=200)

    return pod_train, pod_test
      
    