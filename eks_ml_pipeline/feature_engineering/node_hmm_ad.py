import numpy as np
import pandas as pd
import random
import multiprocessing
from functools import partial
from pyspark.sql.functions import col, count
from sklearn.preprocessing import StandardScaler
from ..utilities import feature_processor, null_report, S3Utilities
from ..inputs import feature_engineering_input
from devex_sdk import Pyspark_data_ingestion, get_features
from .train_test_split import all_rectypes_train_test_split


"""
this feature engineering functions will help us run bach jobs that builds training data for Anomaly Detection models
"""
def node_hmm_ad_preprocessing(input_feature_group_name, input_feature_group_version, input_year, input_month, input_day, input_hour, input_setup = "default"):
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
            processed_node_df: pre processed node dataframe

    """

    pyspark_node_hmm_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, setup = input_setup, filter_column_value ='Node')
    err, pyspark_node_hmm_df = pyspark_node_hmm_data.read()

    if err == 'PASS':

        #get features
        features_df = get_features(input_feature_group_name,input_feature_group_version)
        features = features_df["feature_name"].to_list()
        processed_features = feature_processor.cleanup(features)

        model_parameters = features_df["model_parameters"].iloc[0]
        time_steps = model_parameters["time_steps"]

        #filter inital node df based on request features
        node_hmm_df = pyspark_node_hmm_df.select("Timestamp", "InstanceId", *processed_features)
        node_hmm_df = node_hmm_df.withColumn("Timestamp",(col("Timestamp")/1000).cast("timestamp"))

        # Drop NA
        cleaned_node_hmm_df = node_hmm_df.na.drop(subset=processed_features)

        #Quality(timestamp filtered) nodes
        quality_filtered_node_hmm_df = cleaned_node_hmm_df.groupBy("InstanceId").agg(count("Timestamp").alias("timestamp_count"))
        # to get data that is closer to 1min apart
        quality_filtered_hmm_nodes = quality_filtered_node_hmm_df.filter(col("timestamp_count") >= 2*time_steps)

        #Processed Node DF
        processed_node_hmm_df = cleaned_node_hmm_df.filter(col("InstanceId").isin(quality_filtered_hmm_nodes["InstanceId"]))

        return features_df, processed_node_hmm_df

    else:
        empty_df = pd.DataFrame()
        return empty_df, empty_df


def node_hmm_ad_feature_engineering(instance_id, input_df, input_features, input_scaled_features, input_time_steps):
    """
    inputs
    ------
            instance_id: String
            randomly pick instance id

            input_df: df
            preprocessing and filtered node df

            input_features: list
            list of selected features

            input_scaled_features: list
            list of tobe scaled features

            input_time_steps: int
            model parameter time steps

    outputs
    -------
            node_fe_df: df
            training data df for exposing it as data product

    """

    ##pick random df, and normalize
    node_fe_hmm_df = input_df.loc[(input_df["InstanceId"] == instance_id)]
    node_fe_hmm_df = node_fe_hmm_df.sort_values(by='Timestamp').reset_index(drop=True)
    node_fe_hmm_df_len = len(node_fe_hmm_df)

    #tensor builder
    start = random.choice(range(node_fe_hmm_df_len-input_time_steps))
    node_fe_hmm_df = node_fe_hmm_df[start:start+input_time_steps]

    #scaler transformations
    scaler = StandardScaler()
    node_fe_hmm_df[input_scaled_features] = scaler.fit_transform(node_fe_hmm_df[input_features])

    return node_fe_hmm_df


def node_hmm_list_generator(input_data_type, input_split_ratio, input_node_hmm_df, input_node_features_df):
    """
    inputs
    ------
            input_data_type: String
            builds n_samples based on input string

            input_split_ratio: list
            list of split parameters

            input_node_df: df
            preprocessing and filtered node df

            input_node_features_df: df
            processed node features df

    outputs
    -------
            node_list: list
            randomly selected list of node id's with replacement

            input_node_df: df
            final node df with newly added columns

    """
    model_parameters = input_node_features_df["model_parameters"].iloc[0]
    time_steps = model_parameters["time_steps"]
    hour_params = model_parameters["hour_params"]
    weight = model_parameters["weight"]

    if input_data_type == 'train':
        n_samples = round(hour_params*1440*weight/time_steps)
    elif input_data_type == 'test':
        n_samples = round(hour_params*1440*weight/time_steps)

    input_node_hmm_df['freq'] = input_node_hmm_df.groupby('InstanceId')['InstanceId'].transform('count')
    input_node_hmm_df = input_node_hmm_df[input_node_hmm_df["freq"] > time_steps]

    node_hmm_list = input_node_hmm_df['InstanceId'].sample(n_samples).to_list()

    return node_hmm_list, input_node_hmm_df


def node_hmm_fe_pipeline(feature_group_name, feature_version,
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
    node_hmm_features_data, node_hmm_processed_data = node_hmm_ad_preprocessing(feature_group_name, feature_version, partition_year, partition_month, partition_day, partition_hour, spark_config_setup)

    #parsing model parameters
    scaled_features = []
    model_parameters = node_hmm_features_data["model_parameters"].iloc[0]
    features =  feature_processor.cleanup(node_hmm_features_data["feature_name"].to_list())
    time_steps = model_parameters["time_steps"]

    for feature in features:
        scaled_features = scaled_features + ["scaled_"+feature]

    #test, train split
    node_hmm_train_split = node_hmm_features_data["model_parameters"].iloc[0]["split_ratio"]
    node_hmm_test_split =  round(1 - node_hmm_train_split,2)
    node_hmm_train_data, node_hmm_test_data = all_rectypes_train_test_split(node_hmm_processed_data, [node_hmm_train_split,node_hmm_test_split])

    #converting pyspark df's to pandas df
    node_hmm_train_data = node_hmm_train_data.toPandas()
    node_hmm_test_data = node_hmm_test_data.toPandas()

    #intializing s3 utils
    s3_utils = S3Utilities(bucket,feature_group_name, feature_version)

    #writing df's to s3 bucket
    s3_utils.awswrangler_pandas_dataframe_to_s3(node_hmm_train_data, "data", "pandas_df", f'raw_training_{file_name}.parquet')
    s3_utils.awswrangler_pandas_dataframe_to_s3(node_hmm_test_data, "data", "pandas_df", f'raw_testing_{file_name}.parquet')

    #reading df's from s3 bucket
    node_hmm_train_data = s3_utils.read_parquet_to_pandas_df("data" , "pandas_df", f'raw_training_{file_name}.parquet')
    node_hmm_test_data = s3_utils.read_parquet_to_pandas_df("data" , "pandas_df", f'raw_testing_{file_name}.parquet')

    #generating random selected list of node id's
    selected_hmm_node_train_list, processed_node_hmm_train_data = node_hmm_list_generator( 'train', [node_hmm_train_split,node_hmm_test_split], node_hmm_train_data, node_hmm_features_data)
    selected_hmm_node_test_list, processed_node_hmm_test_data = node_hmm_list_generator( 'test', [node_hmm_train_split,node_hmm_test_split], node_hmm_test_data, node_hmm_features_data)

    #getting number of cores per kernel
    num_cores = multiprocessing.cpu_count()

    #Train data feature engineering
    node_hmm_training_list = multiprocessing.Pool(num_cores).map(partial(node_hmm_ad_feature_engineering,
                         input_df=processed_node_hmm_train_data, input_features=features, input_scaled_features=scaled_features, input_time_steps=time_steps), selected_hmm_node_train_list)
    node_hmm_training_df = pd.concat(node_hmm_training_list)
    s3_utils.awswrangler_pandas_dataframe_to_s3(node_hmm_training_df, "data" , "pandas_df", f'training_{file_name}.parquet')


    #Test data feature engineering
    node_hmm_testing_list = multiprocessing.Pool(num_cores).map(partial(node_hmm_ad_feature_engineering,
                         input_df=processed_node_hmm_test_data, input_features=features, input_scaled_features=scaled_features, input_time_steps=time_steps), selected_hmm_node_test_list)
    node_hmm_testing_df = pd.concat(node_hmm_testing_list)
    s3_utils.awswrangler_pandas_dataframe_to_s3(node_hmm_testing_df, "data" , "pandas_df", f'testing_{file_name}.parquet')

