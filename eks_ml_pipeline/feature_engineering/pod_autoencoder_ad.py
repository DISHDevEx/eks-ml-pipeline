import numpy as np
import random
from ..utilities import feature_processor, null_report
from msspackages import Pyspark_data_ingestion, get_features
from pyspark import StorageLevel
from pyspark.sql import Window
from pyspark.sql.functions import col, count, rand, row_number, get_json_object
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

"""
Contributed by Madhu Bandi, Evgeniya Dontsova and Praveen Mada
MSS Dish 5g - Pattern Detection

this feature engineering functions will help us run bach jobs that builds training data for Anomaly Detection models
"""

def pod_autoencoder_ad_preprocessing(feature_group_name, feature_group_created_date, input_year, input_month, input_day, input_hour, input_setup = "default"):
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
            final_pod_df: pre processed node dataframe
            
    """

    pod_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, setup = input_setup, filter_column_value ='Pod')
    err, pod_df = pod_data.read()
    pod_df = pod_df.persist()
    pod_df = pod_df.select(*pod_df.columns,
                           get_json_object(col("kubernetes"),"$.pod_id").alias("pod_id"),
                           col("pod_status"))

 
    if err == 'PASS':
        #get features
        features_df = get_features(feature_group_name,feature_group_created_date)
        features = features_df["feature_name"].to_list()
        
        processed_features = feature_processor.cleanup(features)
    
        #filter inital pod df based on request features
        pod_df = pod_df.select("Timestamp", "pod_id", "pod_status", *processed_features)
        pod_df = pod_df.withColumn("Timestamp",(col("Timestamp")/1000).cast("timestamp"))
        cleaned_pod_df = pod_df.na.drop(subset=processed_features)
        
        #Quality(timestamp filtered) pods
        cleaned_pod_df = cleaned_pod_df.filter(col("pod_status") == "Running")
        quality_filtered_pod_df = cleaned_pod_df.groupBy("pod_id").agg(count("Timestamp").alias("timestamp_count"))
        quality_filtered_pods = quality_filtered_pod_df.filter(col("timestamp_count").between(45,75))

        #Processed pod DF                                                      
        final_pod_df = cleaned_pod_df.filter(col("pod_id").isin(quality_filtered_pods["pod_id"]))
        final_pod_df = final_pod_df.sort("Timestamp")
        #final_pod_df.show(truncate=False)
        
        # Running pods Only
        #final_pod_df = final_pod_df.filter(final_pod_df.pod_status == "Running")
        
        #Drop duplicates on Pod_ID and Timestamp and keep first
        final_pod_df = final_pod_df.dropDuplicates(['pod_id', 'Timestamp'])
        
        #Drop rows with nans 
        non_null_pod_df = final_pod_df.na.drop("all")
        
        #Null report
        null_report_df = null_report.report_generator(final_pod_df, processed_features)     
        #null_report_df.show(truncate=False)
        
        return features_df, final_pod_df
    else:
        empty_df = pd.DataFrame()
        return empty_df
        


def pod_autoencoder_ad_feature_engineering(input_pod_features_df, input_pod_processed_df):
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

    model_parameters = input_pod_features_df["model_parameters"]
    features =  feature_processor.cleanup(input_pod_features_df["feature_name"].to_list())
    
    time_steps = model_parameters[0]["time_steps"]
    batch_size = model_parameters[0]["batch_size"]
    n_samples = batch_size * model_parameters[0]["sample_multiplier"]
    
    pod_tensor = np.zeros((n_samples,time_steps,len(features)))
    final_pod_fe_df = None
    
    input_pod_processed_df.persist(StorageLevel.MEMORY_ONLY)
    #for n in range(n_samples):
    n = 0;
    while n < n_samples:
        
        ##pick random df, and normalize
        random_instance_id= random.choice(input_pod_processed_df.select("pod_id").rdd.flatMap(list).collect())
        pod_fe_df = input_pod_processed_df[(input_pod_processed_df["pod_id"] == random_instance_id)][["Timestamp", "pod_id"] + features].select('*')
        pod_fe_df = pod_fe_df.sort("Timestamp")
        pod_fe_df = pod_fe_df.na.drop(subset=features)

        #scaler transformations
        assembler = VectorAssembler(inputCols=features, outputCol="vectorized_features")
        scaler = StandardScaler(inputCol = "vectorized_features", outputCol = "scaled_features", withMean=True, withStd=True)
        pipeline = Pipeline(stages=[assembler, scaler])
        pod_fe_df = pipeline.fit(pod_fe_df).transform(pod_fe_df)
        
        #fix negative number bug 
        if pod_fe_df.count()-time_steps <= 0:
            break

        #tensor builder
        start = random.choice(range(pod_fe_df.count()-time_steps))
        pod_tensor_df = pod_fe_df.withColumn('rn', row_number().over(Window.partitionBy("pod_id").orderBy("Timestamp"))).filter((col("rn") >= start) & (col("rn") < start+time_steps)).select("scaled_features")
        pod_tensor_list = pod_tensor_df.select("scaled_features").rdd.flatMap(list).collect()
        
        #fix shape mismatch bug
        if len(pod_tensor_list) == time_steps:
            pod_tensor[n,:,:] = pod_tensor_list

            if not final_pod_fe_df:
                final_pod_fe_df = pod_fe_df
            else:
                final_pod_fe_df = final_pod_fe_df.union(pod_fe_df)
        else:
            #n_samples = n_samples+1
            break
            
        n += 1
 
    final_pod_fe_df = final_pod_fe_df.select("Timestamp","pod_id",*features,"scaled_features")
    
    input_pod_processed_df.unpersist()

    return final_pod_fe_df, pod_tensor

    

def pod_autoencoder_train_test_split(input_df):
    
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
    
    
    pod_train, pod_test = input_df.randomSplit(weights=[0.8,0.2], seed=200)

    return pod_train, pod_test
      
    