import numpy as np
import pandas as pd
import random
from ..utilities import feature_processor, null_report, multiprocessing_runner
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
        
        model_parameters = features_df["model_parameters"].iloc[0]
        time_steps = model_parameters["time_steps"]

        #filter inital node df based on request features
        node_df = pyspark_node_df.select("Timestamp", "InstanceId", *processed_features)
        node_df = node_df.withColumn("Timestamp",(col("Timestamp")/1000).cast("timestamp"))
        
        # Drop NA
        cleaned_node_df = node_df.na.drop(subset=processed_features)

        #Quality(timestamp filtered) nodes
        quality_filtered_node_df = cleaned_node_df.groupBy("InstanceId").agg(count("Timestamp").alias("timestamp_count"))
        # to get data that is closer to 1min apart
        quality_filtered_nodes = quality_filtered_node_df.filter(col("timestamp_count") >= 2*time_steps)
        
        #Processed Node DF                                                      
        processed_node_df = cleaned_node_df.filter(col("InstanceId").isin(quality_filtered_nodes["InstanceId"]))
        
        return features_df, processed_node_df

    else:
        empty_df = pd.DataFrame()
        return empty_df, empty_df

def node_autoencoder_ad_feature_engineering(instance_id, input_df, input_features, input_scaled_features, input_time_steps):
    """
    inputs
    ------
            instance_id: String
            randomly pick instance id
            
            input_node_processed_df: df
            preprocessing and filtered node df 
            
            input_features: list
            list of selected features
            
            input_scaled_features: list
            list of tobe scaled features
            
            input_time_steps: int
            model parameter time steps

    outputs
    -------
            node_fe_df: training data df for exposing it as data product
            
    """

    ##pick random df, and normalize
    node_fe_df = input_df.loc[(input_df["InstanceId"] == instance_id)]
    node_fe_df = node_fe_df.sort_values(by='Timestamp').reset_index(drop=True)
    node_fe_df_len = len(node_fe_df)

    #tensor builder
    start = random.choice(range(node_fe_df_len-input_time_steps))
    node_fe_df = node_fe_df[start:start+input_time_steps]

    #scaler transformations
    scaler = StandardScaler()
    node_fe_df[input_scaled_features] = scaler.fit_transform(node_fe_df[input_features])

    return node_fe_df


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


def node_autoencoder_list_generator(input_data_type, input_split_ratio, input_node_processed_df, input_node_features_df):
    """
    inputs
    ------
            input_data_type: String
            builds n_samples based on input string
            
            input_split_ratio: list
            list of split parameters
            
            input_node_processed_df: df
            preprocessing and filtered node df 
            
            input_node_features_df: df
            processed node features df
 
    outputs
    -------
            node_list: list
            randomly selected list of node id's with replacement
            
    """
    model_parameters = input_node_features_df["model_parameters"].iloc[0]
    time_steps = model_parameters["time_steps"]
    batch_size = model_parameters["batch_size"]

    if input_data_type == 'train':
        n_samples = batch_size * model_parameters["train_sample_multiplier"]
    elif input_data_type == 'test':
        n_samples = round((batch_size * model_parameters["train_sample_multiplier"]* input_split_ratio[1])/ input_split_ratio[0])
        
    
    input_node_df['freq'] = input_node_df.groupby('InstanceId')['InstanceId'].transform('count')
    input_node_df = input_node_df[node_training_df["freq"] > time_steps]
    
    node_list = input_node_df['InstanceId'].sample(n_samples).to_list()
    
    return node_list, input_node_df


def node_autoencoder_fe_runner():
    %%time
from eks_ml_pipeline import node_autoencoder_ad_preprocessing, node_autoencoder_ad_feature_engineering, node_autoencoder_train_test_split, node_autoencoder_list_generator
from eks_ml_pipeline import write_tensor, awswrangler_pandas_dataframe_to_s3, read_parquet_to_pandas_df
import pandas as pd
import numpy as np
import multiprocessing
from functools import partial

#pre processing
node_features_data, node_processed_data = node_autoencoder_ad_preprocessing("node_autoencoder_ad","v0.0.2","2022","9","29",-1,"384gb")

#test, train split
node_train_split = node_features_data["model_parameters"].iloc[0]["split_ratio"]
node_test_split =  round(1 - node_train_split,2)
node_train_data, node_test_data = node_autoencoder_train_test_split(node_processed_data, [node_train_split,node_test_split])

#converting pyspark df's to pandas df
node_train_data = node_train_data.toPandas()
node_test_data = node_test_data.toPandas()

#writing df's to s3 bucket
awswrangler_pandas_dataframe_to_s3(node_train_data,  "dish-5g.core.pd.g.dp.eks.logs.e", "node_autoencoder_ad", "v0.0.2", "raw_training_2022_9_29")
awswrangler_pandas_dataframe_to_s3(node_test_data,  "dish-5g.core.pd.g.dp.eks.logs.e", "node_autoencoder_ad", "v0.0.2", "raw_testing_2022_9_29")

#reading df's from s3 bucket
node_train_data = read_parquet_to_pandas_df("dish-5g.core.pd.g.dp.eks.logs.e", "node_autoencoder_ad", "v0.0.2", "raw_training_2022_9_29")
node_test_data = read_parquet_to_pandas_df("dish-5g.core.pd.g.dp.eks.logs.e", "node_autoencoder_ad", "v0.0.2", "raw_testing_2022_9_29")

#generating random selected list of node id's
selected_node_train_list, processed_node_train_data = node_autoencoder_list_generator( 'train', [node_train_split,node_test_split], node_train_data, node_features_data)
selected_node_test_list, processed_node_test_data = node_autoencoder_list_generator( 'test', [node_train_split,node_test_split], node_test_data, node_features_data)


model_parameters = node_features_data["model_parameters"].iloc[0]
features =  feature_processor.cleanup(node_features_data["feature_name"].to_list())
time_steps = model_parameters["time_steps"]

scaled_features = []
for feature in features:
    scaled_features = scaled_features + ["scaled_"+feature]


num_cores = multiprocessing.cpu_count()
print(num_cores)

#Train data feature engineering
node_training_list = multiprocessing.Pool(num_cores).map(partial(node_autoencoder_ad_feature_engineering, 
                     input_df=processed_node_train_data, input_features=features, input_scaled_features=scaled_features, input_time_steps=time_steps), selected_node_train_list)
node_training_df = pd.concat(node_training_list)
node_training_tensor = np.array(list(map(lambda x: x.to_numpy(), node_training_list)))
write_tensor(node_training_tensor, "dish-5g.core.pd.g.dp.eks.logs.e", "node_autoencoder_ad", "v0.0.2", "training_2022_9_29")
awswrangler_pandas_dataframe_to_s3(node_training_df,  "dish-5g.core.pd.g.dp.eks.logs.e", "node_autoencoder_ad", "v0.0.2", "training_2022_9_29")


#Test data feature engineering
node_testing_list = multiprocessing.Pool(num_cores).map(partial(node_autoencoder_ad_feature_engineering, 
                     input_df=processed_node_test_data, input_features=features, input_scaled_features=scaled_features, input_time_steps=time_steps), selected_node_test_list)
node_testing_df = pd.concat(node_testing_list)
node_testing_tensor = np.array(list(map(lambda x: x.to_numpy(), node_testing_list)))
write_tensor(node_testing_tensor, "dish-5g.core.pd.g.dp.eks.logs.e", "node_autoencoder_ad", "v0.0.2", "testing_2022_9_29")
awswrangler_pandas_dataframe_to_s3(node_testing_df,  "dish-5g.core.pd.g.dp.eks.logs.e", "node_autoencoder_ad", "v0.0.2", "testing_2022_9_29")