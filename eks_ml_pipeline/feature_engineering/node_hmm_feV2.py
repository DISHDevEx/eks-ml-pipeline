
import numpy as np
import pandas as pd
import random
import multiprocessing
from functools import partial
from pyspark.sql.functions import col, count
from sklearn.preprocessing import StandardScaler
from ..utilities import feature_processor, null_report, S3Utilities
from ..inputs import feature_engineering_input
from msspackages import Pyspark_data_ingestion, get_features
from .train_test_split import all_rectypes_train_test_split



"""
Contributed by Ruyi Yang
MSS Dish 5g - Pattern Detection

this feature engineering functions will help us run bach jobs that builds training data for Anomaly Detection models
"""

def node_hmm_fe_v2(feature_group_name, feature_group_version, input_year, input_month, input_day, input_hour, input_setup = "default"):
    
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

    node_data = Pyspark_data_ingestion(
        year = input_year, 
        month = input_month, 
        day = input_day, 
        hour = input_hour, 
        setup = input_setup, 
        filter_column_value ='Node')
    err, node_df = node_data.read()
    # node_df = node_df.select("InstanceId",'Timestamp','node_cpu_utilization','node_memory_utilization')

 
    if err == 'PASS':
        
        #get features
        features_df = get_features(feature_group_name,feature_group_version)
        features = features_df["feature_name"].to_list()
        processed_features = feature_processor.cleanup(features)
        
        model_parameters = features_df["model_parameters"].iloc[0]
  
        #drop na values in node cpu and memory utilization
        node_df = node_df.select("InstanceId","Timestamp", *processed_features)
        node_df = node_df.na.drop(subset=processed_features)
        
        #remove nodes which has a time gap over 2 minutes (epochtime = 2*60*1000=120000)
        w = Window.partitionBy('InstanceId').orderBy('Timestamp')
        node_df = node_df.withColumn('lead', f.lag('Timestamp', 1).over(w)) \
              .withColumn(
                'Timediff', 
                f.when(f.col('lead').isNotNull(), 
                f.col('Timestamp') - f.col('lead'))
                .otherwise(f.lit(None)))
               
        
        temp_df = node_df\
            .groupby("InstanceId")\
            .max("Timediff")\
            .select('InstanceId',f.col('max(TimeDiff)').alias('maxDiff'))\
            .filter("maxDiff<=120000")
                                                             
        node_df = node_df.filter(col("InstanceId").isin(temp_df['InstanceId']))
        node_df = node_df.sort("InstanceId","Timestamp")
        node_df = node_df.select('InstanceId','Timestamp',*features)
        
        #Drop rows with nans 
        node_df = node_df.na.drop("all")
           
        
        return features_df, node_df
    else:
        empty_df = pd.DataFrame()
        return empty_df
    
    
def node_hmm_train_test_split(input_df,split = 0.5):
    
    """
    inputs
    ------
            
            input_df: df
            preprocessing node df 
            
            weight: float
            select weight of split
            
    
    outputs
    -------
            
            pod_train: df
            training data df for exposing it as data product
            
            
            pod_test: df
            testing data df for exposing it as data product
            
    """
    temp_df = input_df.select('InstanceId')
    node_train_id, node_test_id = temp_df.randomSplit(weights=[split,1-split], seed=200)  
    node_train = input_df.filter(col("InstanceId").isin(node_train_id['InstanceId']))
    node_test = input_df.filter(col("InstanceId").isin(node_test_id['InstanceId']))
    
    return node_train, node_test

def feature_engineering(input_df, features):
    
    """
    inputs
    ------
            instance_id: String
            randomly pick instance id
            
            input_df: df
            preprocessing node df 
            
            
            
    outputs
    -------
            
            features_list: list
            nested list in which each element is a 2-D nparray
            
    """
    
    #sort data
    input_df = input_df.sort_values(by = ['InstanceId','Timestamp'])  
    scaled_features_list = []
    
    #standardize data by instanceid and return the nested list of nparray
    instance_list = input_df['InstanceId'].unique()
    for i in instance_list:
        sub = input_df.loc[input_df.InstanceId == i]
        scaled_features_list.append(scaler.fit_transform(sub[features]))
    
    return scaled_features_list


def node_fe_pipeline(feature_group_name, feature_version,
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
    node_features_data, node_processed_data = node_hmm_fe_v2(feature_group_name, feature_group_version, input_year, input_month, input_day, input_hour, input_setup)
    
    #parsing model parameters
    scaled_features = []
    model_parameters = node_features_data["model_parameters"].iloc[0]
    features =  feature_processor.cleanup(node_features_data["feature_name"].to_list())
    for feature in features:
        scaled_features = scaled_features + ["scaled_"+feature]
        
    #train, test split
    node_train_data, node_test_data = node_hmm_train_test_split(node_processed_data,split = 0.5)
    
    # convert pyspark's df to pandas df
    node_train_data = node_train_data.toPandas()
    node_test_data = node_test_data.toPandas()
    
    #intializing s3 utils
    s3_utils = S3Utilities(bucket,feature_group_name, feature_version)
    
    #writing df's to s3 bucket
    s3_utils.awswrangler_pandas_dataframe_to_s3(node_train_data, "data", "pandas_df", f'raw_training_{file_name}.parquet')
    s3_utils.awswrangler_pandas_dataframe_to_s3(node_test_data, "data", "pandas_df", f'raw_testing_{file_name}.parquet')
    
    #reading df's from s3 bucket
    node_train_data = s3_utils.read_parquet_to_pandas_df("data" , "pandas_df", f'raw_training_{file_name}.parquet')
    node_test_data = s3_utils.read_parquet_to_pandas_df("data" , "pandas_df", f'raw_testing_{file_name}.parquet')

    #getting number of cores per kernel
    num_cores = multiprocessing.cpu_count()
    
    #Train data feature engineering
    node_training_df = feature_engineering(node_train_data, features)
    s3_utils.awswrangler_pandas_dataframe_to_s3(node_training_df,  "data" , "pandas_df", f'testing_{file_name}.parquet')
    
    #Test data feature engineering
    node_testing_df = feature_engineering(node_test_data, features)
    s3_utils.awswrangler_pandas_dataframe_to_s3(node_testing_df,  "data" , "pandas_df", f'testing_{file_name}.parquet')
