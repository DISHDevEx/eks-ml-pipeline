import numpy as np
import random
from utilities import feature_processor, null_report
from msspackages import Pyspark_data_ingestion, get_features
from pyspark.sql import Window
from pyspark.sql.functions import col, count, rand, row_number, lit
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline


def node_autoencoder_ad_preprocessing(feature_group_name, feature_group_created_date, input_year, input_month, input_day, input_hour, input_setup = "default"):

    pyspark_node_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, setup = input_setup, filter_column_value ='Node')
    err, pyspark_node_df = pyspark_node_data.read()

    if err == 'PASS':

        #get features
        features_df = get_features(feature_group_name,feature_group_created_date)
        features = features_df["feature_name"].to_list()
        processed_features = feature_processor.cleanup(features)
    
        #filter inital node df based on request features
        node_df = pyspark_node_df.select("Timestamp", "InstanceId", *processed_features)
        node_df = node_df.withColumn("Timestamp",(col("Timestamp")/1000).cast("timestamp"))
        
        # Drop NA
        cleaned_node_df = node_df.na.drop(subset=processed_features)

        
        #Quality(timestamp filtered) nodes
        quality_filtered_node_df = cleaned_node_df.groupBy("InstanceId").agg(count("Timestamp").alias("timestamp_count"))
        quality_filtered_nodes = quality_filtered_node_df.filter(col("timestamp_count").between(45,75))
        

        #Processed Node DF                                                      
        processed_node_df = cleaned_node_df.filter(col("InstanceId").isin(quality_filtered_nodes["InstanceId"]))
        
        #Null report
        null_report_df = null_report.report_generator(processed_node_df, processed_features)     
        #null_report_df.show(truncate=False)
        
        return features_df, processed_node_df

    else:
        empty_df = pd.DataFrame()
        return empty_df, empty_df
    
    

def node_autoencoder_ad_feature_engineering(input_node_features_df, input_node_processed_df):

    model_parameters = input_node_features_df["model_parameters"]
    features =  feature_processor.cleanup(input_node_features_df["feature_name"].to_list())
    
    time_steps = model_parameters[0]["time_steps"]
    batch_size = model_parameters[0]["batch_size"]
    n_samples = batch_size * model_parameters[0]["sample_multiplier"]
    
    node_data = np.zeros((n_samples,time_steps,len(features)))
    for n in range(n_samples):
        ##pick random df, and normalize
        random_instance_df= input_node_processed_df.select("InstanceId").orderBy(rand()).limit(1)
        node_fe_df = input_node_processed_df[(input_node_processed_df["InstanceId"] == random_instance_df.first()["InstanceId"])][["Timestamp", "InstanceId"] + features].select('*')
        node_fe_df = node_fe_df.sort("Timestamp")
        
        #scaler transformations
        assembler = VectorAssembler(inputCols=features, outputCol="vectorized_features")
        scaler = StandardScaler(inputCol = "vectorized_features", outputCol = "scaled_features", withMean=True, withStd=True)
        pipeline = Pipeline(stages=[assembler, scaler])
        node_fe_df = pipeline.fit(node_fe_df).transform(node_fe_df)
            
    node_fe_df = node_fe_df.select("Timestamp","InstanceId",*features,"scaled_features")
    node_fe_df.show(truncate=False) 

    return node_fe_df



def node_autoencoder_train_test_split(input_df):
    
    node_train, node_test = input_df.randomSplit(weights=[0.8,0.2], seed=200)

    return node_train, node_test
    