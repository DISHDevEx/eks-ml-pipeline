import numpy as np
import random
from utilities import feature_processor, null_report
from msspackages import Pyspark_data_ingestion, get_features
from pyspark.sql import Window
from pyspark.sql.functions import get_json_object, col, count, rand, row_number, concat_ws
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

"""
Contributed by David Cherney and Praveen Mada
MSS Dish 5g - Pattern Detection

this feature engineering functions will help us run bach jobs that builds training data for Anomaly Detection models
"""

def container_autoencoder_ad_preprocessing(feature_group_name, feature_group_created_date, input_year, input_month, input_day, input_hour, input_setup = "default"):
    """
    inputs
    ------
            feature_group_name: STRING
            json name to get the required features
            
            feature_group_created_date: STRING
            json created date to get the latest features 
            
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
        features_df = get_features(feature_group_name,feature_group_created_date)
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
        
        #Null report
        null_report_df = null_report.report_generator(processed_container_df, processed_features)
        
        return features_df, processed_container_df

    else:
        empty_df = pd.DataFrame()
        return empty_df, empty_df
 
    
def container_autoencoder_ad_feature_engineering(input_container_features_df, input_container_processed_df):
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
    
    model_parameters = input_container_features_df["model_parameters"]
    features =  feature_processor.cleanup(input_container_features_df["feature_name"].to_list())
    
    time_steps = model_parameters[0]["time_steps"]
    batch_size = model_parameters[0]["batch_size"]
    n_samples = batch_size * model_parameters[0]["sample_multiplier"]
    
    container_tensor = np.zeros((n_samples,time_steps,len(features)))
    final_container_fe_df = None

    for n in range(n_samples):
        ##pick random df, and normalize
        random_container_id= random.choice(input_container_processed_df.select("container_name_pod_id").rdd.flatMap(list).collect())
        container_fe_df = input_container_processed_df[(input_container_processed_df["container_name_pod_id"] == random_container_id)][["Timestamp","container_name_pod_id"] + features].select('*')
        container_fe_df = container_fe_df.sort("Timestamp")
        container_fe_df = container_fe_df.na.drop(subset=features)
        
        #scaler transformations
        assembler = VectorAssembler(inputCols=features, outputCol="vectorized_features")
        scaler = StandardScaler(inputCol = "vectorized_features", outputCol = "scaled_features", withMean=True, withStd=True)
        pipeline = Pipeline(stages=[assembler, scaler])
        container_fe_df = pipeline.fit(container_fe_df).transform(container_fe_df)
        
        #tensor builder
        start = random.choice(range(container_fe_df.count()-time_steps))
        container_tensor_df = container_fe_df.withColumn("rn", row_number().over(Window.orderBy("container_name_pod_id"))).filter((col("rn") >= start) & (col("rn") < start+time_steps)).select("scaled_features")
        container_tensor_list = container_tensor_df.select("scaled_features").rdd.flatMap(list).collect()
        if len(container_tensor_list) == time_steps:
            container_tensor[n,:,:] = container_tensor_list

            if not final_container_fe_df:
                final_container_fe_df = container_fe_df
            else:
                final_container_fe_df = final_container_fe_df.union(container_fe_df)

        else:
            n_samples = n_samples+1

    final_container_fe_df = final_container_fe_df.select("Timestamp","container_name_pod_id",*features,"scaled_features")

    return final_container_fe_df, container_tensor

    
def container_autoencoder_train_test_split(input_df):
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
    
    container_train, container_test = input_df.randomSplit(weights=[0.8,0.2], seed=200)

    return container_train, container_test