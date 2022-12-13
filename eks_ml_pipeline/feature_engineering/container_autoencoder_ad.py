import numpy as np
import pandas as pd
import random
from ..utilities import feature_processor, null_report
from msspackages import Pyspark_data_ingestion, get_features
from pyspark.sql.functions import get_json_object, col, count, concat_ws
from sklearn.preprocessing import StandardScaler


"""
Contributed by David Cherney and Praveen Mada
MSS Dish 5g - Pattern Detection

this feature engineering functions will help us run bach jobs that builds training data for Anomaly Detection models
"""

def container_autoencoder_ad_preprocessing(feature_group_name, feature_group_version, input_year, input_month, input_day, input_hour, input_setup = "default"):
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
            processed_container_df: pre processed container dataframe
            
    """

    pyspark_container_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, setup = input_setup,  filter_column_value ='Container')
    err, pyspark_container_df = pyspark_container_data.read()

    if err == 'PASS':
        #get features
        features_df = get_features(feature_group_name,feature_group_version)
        features = features_df["feature_name"].to_list()
        processed_features = feature_processor.cleanup(features)
    
        #filter inital container df based on request features
        container_df = pyspark_container_df.select("Timestamp", concat_ws("-", get_json_object(col("kubernetes"),"$.container_name"), get_json_object(col("kubernetes"),"$.pod_id")).alias("container_name_pod_id"), *processed_features)
        container_df = container_df.withColumn("Timestamp",(col("Timestamp")/1000).cast("timestamp"))
        
        # Drop NA
        cleaned_container_df = container_df.na.drop(subset=processed_features)

        #Quality(timestamp filtered) nodes
        quality_filtered_container_df = cleaned_container_df.groupBy("container_name_pod_id").agg(count("Timestamp").alias("timestamp_count"))
        # to get data that is closer to 1min apart
        quality_filtered_containers = quality_filtered_container_df.filter(col("timestamp_count").between(45,75))
        
        #Processed Container DF                                                      
        processed_container_df = cleaned_container_df.filter(col("container_name_pod_id").isin(quality_filtered_containers["container_name_pod_id"]))
        
        return features_df, processed_container_df

    else:
        empty_df = pd.DataFrame()
        return empty_df, empty_df
 
    
def container_autoencoder_ad_feature_engineering(input_data_type, input_split_ratio, input_container_features_df, input_container_processed_df):
    """
    inputs
    ------
            input_container_features_df: df
            processed conatiner features df
            
            input_container_processed_df: df
            preprocessing and filtered container df 
    
    outputs
    -------
            container_tensor : np array for training the model
            final_container_fe_df: training data df for exposing it as data product
            
    """
    
    model_parameters = input_container_features_df["model_parameters"].iloc[0]
    features =  feature_processor.cleanup(input_container_features_df["feature_name"].to_list())

    time_steps = model_parameters["time_steps"]
    batch_size = model_parameters["batch_size"]

    if input_data_type == 'train':
        n_samples = batch_size * model_parameters["train_sample_multiplier"]
    elif input_data_type == 'test':
         n_samples = round((batch_size * model_parameters["train_sample_multiplier"]* input_split_ratio[1])/ input_split_ratio[0])

    container_tensor = np.zeros((n_samples,time_steps,len(features)))
    final_container_fe_df = pd.DataFrame()
    
    scaled_features = []
    for feature in features:
        scaled_features = scaled_features + ["scaled_"+feature]

    #To Pandas
    input_container_processed_df = input_container_processed_df.toPandas()

    n = 0
    while n < n_samples:
        random_container_id = random.choice(input_container_processed_df["container_name_pod_id"].unique())
        container_fe_df = input_container_processed_df.loc[(input_container_processed_df["container_name_pod_id"] == random_container_id)]
        container_fe_df = container_fe_df.sort_values(by='Timestamp').reset_index(drop=True)
        
        #scaler transformations
        scaler = StandardScaler()
        container_fe_df[scaled_features] = scaler.fit_transform(container_fe_df[features])
        
        container_fe_df_len = len(container_fe_df)
        
        #fix negative number bug 
        if container_fe_df_len-time_steps <= 0:
            print(f'Exception occurred: container_fe_df_len-time_steps = {container_fe_df_len-time_steps}')
            continue
        
        #tensor builder
        start = random.choice(range(container_fe_df_len-time_steps))
        container_tensor[n,:,:] = container_fe_df[start:start+time_steps][scaled_features]

        if final_container_fe_df.empty:
            final_container_fe_df = container_fe_df
        else:
            final_container_fe_df.append(container_fe_df, ignore_index =True)

        print(f'Finished with sample #{n}')

        n +=1


    return final_container_fe_df, container_tensor

    
def container_autoencoder_train_test_split(input_df, split_weights):
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
    
    container_train, container_test = input_df.randomSplit(weights=split_weights, seed=200)

    return container_train, container_test