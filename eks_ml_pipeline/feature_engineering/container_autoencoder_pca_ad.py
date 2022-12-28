import numpy as np
import pandas as pd
import random
import multiprocessing
from functools import partial
from pyspark.sql.functions import get_json_object, col, count, concat_ws
from sklearn.preprocessing import StandardScaler
from ..utilities import feature_processor, null_report, s3_utils
from ..inputs import feature_engineering_input
from msspackages import Pyspark_data_ingestion, get_features


"""
Contributed by David Cherney and Praveen Mada
MSS Dish 5g - Pattern Detection

this feature engineering functions will help us run bach jobs that builds training data for Anomaly Detection models
"""    
def container_train_test_split(input_df, split_weights):
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


def container_ad_preprocessing(input_feature_group_name, input_feature_group_version, input_year, input_month, input_day, input_hour, input_setup = "default"):
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
            processed_container_df: pre processed container dataframe
            
    """

    pyspark_container_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, setup = input_setup,  filter_column_value ='Container')
    err, pyspark_container_df = pyspark_container_data.read()

    if err == 'PASS':
        #get features
        features_df = get_features(feature_group_name,feature_group_version)
        features = features_df["feature_name"].to_list()
        processed_features = feature_processor.cleanup(features)
        
        model_parameters = features_df["model_parameters"].iloc[0]
        time_steps = model_parameters["time_steps"]
    
        #filter inital container df based on request features
        container_df = pyspark_container_df.select("Timestamp", concat_ws("-", get_json_object(col("kubernetes"),"$.container_name"), get_json_object(col("kubernetes"),"$.pod_id")).alias("container_name_pod_id"), *processed_features)
        container_df = container_df.withColumn("Timestamp",(col("Timestamp")/1000).cast("timestamp"))
        
        # Drop NA
        cleaned_container_df = container_df.na.drop(subset=processed_features)

        #Quality(timestamp filtered) nodes
        quality_filtered_container_df = cleaned_container_df.groupBy("container_name_pod_id").agg(count("Timestamp").alias("timestamp_count"))
        # to get data that is closer to 1min apart
        quality_filtered_containers = quality_filtered_container_df.filter(col("timestamp_count") >= 2*time_steps)
        
        #Processed Container DF                                                      
        processed_container_df = cleaned_container_df.filter(col("container_name_pod_id").isin(quality_filtered_containers["container_name_pod_id"]))
        
        return features_df, processed_container_df

    else:
        empty_df = pd.DataFrame()
        return empty_df, empty_df


def container_ad_feature_engineering(container_id, input_df, input_features, input_scaled_features, input_time_steps):
    """
    inputs
    ------
            container_id: String
            randomly pick container id
            
            input_df: df
            preprocessing and filtered container df 
            
            input_features: list
            list of selected features
            
            input_scaled_features: list
            list of tobe scaled features
            
            input_time_steps: int
            model parameter time steps

    outputs
    -------
            container_fe_df: df
            training data df for exposing it as data product
            
    """

    ##pick random df, and normalize
    container_fe_df = input_df.loc[(input_df["container_name_pod_id"] == container_id)]
    container_fe_df = container_fe_df.sort_values(by='Timestamp').reset_index(drop=True)
    container_fe_df_len = len(container_fe_df)

    #tensor builder
    start = random.choice(range(container_fe_df_len-input_time_steps))
    container_fe_df = container_fe_df[start:start+input_time_steps]

    #scaler transformations
    scaler = StandardScaler()
    container_fe_df[input_scaled_features] = scaler.fit_transform(container_fe_df[input_features])

    return container_fe_df


def container_list_generator(input_data_type, input_split_ratio, input_container_df, input_container_features_df):
    """
    inputs
    ------
            input_data_type: String
            builds n_samples based on input string
            
            input_split_ratio: list
            list of split parameters
            
            input_container_df: df
            preprocessing and filtered container df 
            
            input_container_features_df: df
            processed container features df
 
    outputs
    -------
            container_list: list
            randomly selected list of container id's with replacement
            
            input_container_df: df
            final container df with newly added columns
            
    """
    model_parameters = input_container_features_df["model_parameters"].iloc[0]
    time_steps = model_parameters["time_steps"]
    batch_size = model_parameters["batch_size"]

    if input_data_type == 'train':
        n_samples = batch_size * model_parameters["train_sample_multiplier"]
    elif input_data_type == 'test':
        n_samples = round((batch_size * model_parameters["train_sample_multiplier"]* input_split_ratio[1])/ input_split_ratio[0])
        
    
    input_container_df['freq'] = input_container_df.groupby('container_name_pod_id')['container_name_pod_id'].transform('count')
    input_container_df = input_container_df[input_container_df["freq"] > time_steps]
    
    container_list = input_container_df['container_name_pod_id'].sample(n_samples).to_list()
    
    return container_list, input_container_df


def container_fe_pipeline(feature_group_name, feature_version,
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
    container_features_data, container_processed_data = container_ad_preprocessing(feature_group_name, feature_version, partition_year, partition_month, partition_day, partition_hour, spark_config_setup)
    
    #parsing model parameters
    scaled_features = []
    model_parameters = container_features_data["model_parameters"].iloc[0]
    features =  feature_processor.cleanup(container_features_data["feature_name"].to_list())
    time_steps = model_parameters["time_steps"]
    for feature in features:
        scaled_features = scaled_features + ["scaled_"+feature]

    #test, train split
    container_train_split = container_features_data["model_parameters"].iloc[0]["split_ratio"]
    container_test_split =  round(1 - container_train_split,2)
    container_train_data, container_test_data = container_train_test_split(container_processed_data, [container_train_split,container_test_split])

    #converting pyspark df's to pandas df
    container_train_data = container_train_data.toPandas()
    container_test_data = container_test_data.toPandas()

    #writing df's to s3 bucket
    awswrangler_pandas_dataframe_to_s3(container_train_data, bucket , feature_group_name, feature_version, f'raw_training_{file_name}')
    awswrangler_pandas_dataframe_to_s3(container_test_data, bucket , feature_group_name, feature_version, f'raw_testing_{file_name}')

    #reading df's from s3 bucket
    container_train_data = read_parquet_to_pandas_df(bucket , feature_group_name, feature_version, f'raw_training_{file_name}')
    container_test_data = read_parquet_to_pandas_df(bucket , feature_group_name, feature_version, f'raw_testing_{file_name}')

    #generating random selected list of container id's
    selected_container_train_list, processed_container_train_data = container_list_generator( 'train', [container_train_split,container_test_split], container_train_data, container_features_data)
    selected_container_test_list, processed_container_test_data = container_list_generator( 'test', [container_train_split,container_test_split], container_test_data, container_features_data)

    num_cores = multiprocessing.cpu_count()
    print(num_cores)

    #Train data feature engineering
    container_training_list = multiprocessing.Pool(num_cores).map(partial(container_ad_feature_engineering, 
                         input_df=processed_container_train_data, input_features=features, input_scaled_features=scaled_features, input_time_steps=time_steps), selected_container_train_list)
    container_training_df = pd.concat(container_training_list)
    container_training_tensor = np.array(list(map(lambda x: x.to_numpy(), container_training_list)))
    write_tensor(container_training_tensor, bucket , feature_group_name, feature_version, f'training_{file_name}')
    awswrangler_pandas_dataframe_to_s3(container_training_df, bucket , feature_group_name, feature_version, f'training_{file_name}')


    #Test data feature engineering
    container_testing_list = multiprocessing.Pool(num_cores).map(partial(container_ad_feature_engineering, 
                         input_df=processed_container_test_data, input_features=features, input_scaled_features=scaled_features, input_time_steps=time_steps), selected_container_test_list)
    container_testing_df = pd.concat(container_testing_list)
    container_testing_tensor = np.array(list(map(lambda x: x.to_numpy(), container_testing_list)))
    write_tensor(container_testing_tensor, bucket , feature_group_name, feature_version, f'testing_{file_name}')
    awswrangler_pandas_dataframe_to_s3(container_testing_df,  bucket , feature_group_name, feature_version, f'testing_{file_name}')
    

if __name__ == "__main__":
    #build and save container autoencoder training data to s3
    container_fe_pipeline(*container_autoencoder_fe_input())

    #build and save container pca training data to s3
    container_fe_pipeline(*container_pca_fe_input())   

   