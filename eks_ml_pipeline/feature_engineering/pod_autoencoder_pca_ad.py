import numpy as np
import pandas as pd
import random
import multiprocessing
from functools import partial
from pyspark.sql.functions import col, count, get_json_object
from sklearn.preprocessing import StandardScaler
from ..utilities import feature_processor, null_report, S3Utilities
from ..inputs import feature_engineering_input
from devex_sdk import EKS_Connector, get_features
from .train_test_split import all_rectypes_train_test_split

"""

this feature engineering functions will help us run bach jobs that builds training data for Anomaly Detection models
"""
def pod_ad_preprocessing(input_feature_group_name, input_feature_group_version, input_year, input_month, input_day, input_hour, input_setup = "default"):
    """
    inputs
    ------
            input_feature_group_name: STRING
            json name to get the required features
            
            input_feature_group_version: STRING
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
            final_pod_df: pre processed pod dataframe
            
    """

    pod_data = EKS_Connector(year = input_year, month = input_month, day = input_day, hour = input_hour, setup = input_setup, filter_column_value ='Pod')
    err, pod_df = pod_data.read()

    if err == 'PASS':
        #get features
        features_df = get_features(input_feature_group_name,input_feature_group_version)
        features = features_df["feature_name"].to_list()
        processed_features = feature_processor.cleanup(features)
        
        model_parameters = features_df["model_parameters"].iloc[0]
        time_steps = model_parameters["time_steps"]

        #parsing json column to extract pod_id
        pod_df = pod_df.select("Timestamp",
                           get_json_object(col("kubernetes"),"$.pod_id").alias("pod_id"),
                           "pod_status", *processed_features)
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



def pod_ad_feature_engineering(pod_id, input_df, input_features, input_scaled_features, input_time_steps):
    """
    inputs
    ------
            pod_id: String
            randomly pick pod id
            
            input_df: df
            preprocessing and filtered pod df 
            
            input_features: list
            list of selected features
            
            input_scaled_features: list
            list of tobe scaled features
            
            input_time_steps: int
            model parameter time steps

    outputs
    -------
            pod_fe_df: df
            training data df for exposing it as data product
            
    """

    ##pick random df, and normalize
    pod_fe_df = input_df.loc[(input_df["pod_id"] == pod_id)]
    pod_fe_df = pod_fe_df.sort_values(by='Timestamp').reset_index(drop=True)
    pod_fe_df_len = len(pod_fe_df)

    #tensor builder
    start = random.choice(range(pod_fe_df_len-input_time_steps))
    pod_fe_df = pod_fe_df[start:start+input_time_steps]

    #scaler transformations
    scaler = StandardScaler()
    pod_fe_df[input_scaled_features] = scaler.fit_transform(pod_fe_df[input_features])

    return pod_fe_df


def pod_list_generator(input_data_type, input_split_ratio, input_pod_df, input_pod_features_df):
    """
    inputs
    ------
            input_data_type: String
            builds n_samples based on input string
            
            input_split_ratio: list
            list of split parameters
            
            input_pod_df: df
            preprocessing and filtered pod df 
            
            input_pod_features_df: df
            processed pod features df
 
    outputs
    -------
            pod_list: list
            randomly selected list of pod id's with replacement
            
            input_pod_df: df
            final pod df with newly added columns
            
    """
    model_parameters = input_pod_features_df["model_parameters"].iloc[0]
    time_steps = model_parameters["time_steps"]
    batch_size = model_parameters["batch_size"]

    if input_data_type == 'train':
        n_samples = batch_size * model_parameters["train_sample_multiplier"]
    elif input_data_type == 'test':
        n_samples = round((batch_size * model_parameters["train_sample_multiplier"]* input_split_ratio[1])/ input_split_ratio[0])

    input_pod_df['freq'] = input_pod_df.groupby('pod_id')['pod_id'].transform('count')
    input_pod_df = input_pod_df[input_pod_df["freq"] > time_steps]
    
    pod_list = input_pod_df['pod_id'].sample(n_samples).to_list()
    
    return pod_list, input_pod_df


def pod_fe_pipeline(feature_group_name, feature_version,
            partition_year, partition_month, partition_day,
            partition_hour, spark_config_setup,
            bucket):
    
    ##building file name dynamically
    if partition_hour == -1:
        file_name = f'{partition_year}_{partition_month}_{partition_day}'
    elif partition_day == -1:
        file_name = f'{partition_year}_{partition_month}'
    else:
        file_name = f'{partition_year}_{partition_month}_{partition_day}_{partition_hour}'
    
    #pre processing
    pod_features_data, pod_processed_data = pod_ad_preprocessing(feature_group_name, feature_version, partition_year, partition_month, partition_day, partition_hour, spark_config_setup)
    
    #parsing model parameters
    scaled_features = []
    model_parameters = pod_features_data["model_parameters"].iloc[0]
    features =  feature_processor.cleanup(pod_features_data["feature_name"].to_list())
    time_steps = model_parameters["time_steps"]
    for feature in features:
        scaled_features = scaled_features + ["scaled_"+feature]

    #test, train split
    pod_train_split = pod_features_data["model_parameters"].iloc[0]["split_ratio"]
    pod_test_split =  round(1 - pod_train_split,2)
    pod_train_data, pod_test_data = all_rectypes_train_test_split(pod_processed_data, [pod_train_split,pod_test_split])

    #converting pyspark df's to pandas df
    pod_train_data = pod_train_data.toPandas()
    pod_test_data = pod_test_data.toPandas()

    #intializing s3 utils
    s3_utils = S3Utilities(bucket,feature_group_name, feature_version)

    #writing df's to s3 bucket
    s3_utils.awswrangler_pandas_dataframe_to_s3(pod_train_data, "data", "pandas_df", f'raw_training_{file_name}.parquet')
    s3_utils.awswrangler_pandas_dataframe_to_s3(pod_test_data, "data", "pandas_df", f'raw_testing_{file_name}.parquet')

    #reading df's from s3 bucket
    pod_train_data = s3_utils.read_parquet_to_pandas_df("data" , "pandas_df", f'raw_training_{file_name}.parquet')
    pod_test_data = s3_utils.read_parquet_to_pandas_df("data" , "pandas_df", f'raw_testing_{file_name}.parquet')

    #generating random selected list of pod id's
    selected_pod_train_list, processed_pod_train_data = pod_list_generator( 'train', [pod_train_split,pod_test_split], pod_train_data, pod_features_data)
    selected_pod_test_list, processed_pod_test_data = pod_list_generator( 'test', [pod_train_split,pod_test_split], pod_test_data, pod_features_data)
    
    #getting number of cores per kernel
    num_cores = multiprocessing.cpu_count()

    #Train data feature engineering
    pod_training_list = multiprocessing.Pool(num_cores).map(partial(pod_ad_feature_engineering, 
                         input_df=processed_pod_train_data, input_features=features, input_scaled_features=scaled_features, input_time_steps=time_steps), selected_pod_train_list)
    pod_training_df = pd.concat(pod_training_list)
    pod_training_tensor = np.array(list(map(lambda x: x.to_numpy(), pod_training_list)))
    pod_training_tensor = pod_training_tensor[:,:,-len(scaled_features):]
    s3_utils.write_tensor(pod_training_tensor, "data" , "tensors", f'training_{file_name}.npy')
    s3_utils.awswrangler_pandas_dataframe_to_s3(pod_training_df, "data" , "pandas_df", f'training_{file_name}.parquet')

    #Test data feature engineering
    pod_testing_list = multiprocessing.Pool(num_cores).map(partial(pod_ad_feature_engineering, 
                         input_df=processed_pod_test_data, input_features=features, input_scaled_features=scaled_features, input_time_steps=time_steps), selected_pod_test_list)
    pod_testing_df = pd.concat(pod_testing_list)
    pod_testing_tensor = np.array(list(map(lambda x: x.to_numpy(), pod_testing_list)))
    pod_testing_tensor = pod_testing_tensor[:,:,-len(scaled_features):]
    s3_utils.write_tensor(pod_testing_tensor, "data" , "tensors", f'testing_{file_name}.npy')
    s3_utils.awswrangler_pandas_dataframe_to_s3(pod_testing_df, "data" , "pandas_df", f'testing_{file_name}.parquet')
  

      
    