import numpy as np
import random
from utilities import feature_processor, null_report
from msspackages import Pyspark_data_ingestion, get_features
from pyspark.sql.functions import col, count, rand, get_json_object
from pyspark.ml.feature import VectorAssembler, StandardScaler


def pod_autoencoder_ad_preprocessing(feature_group_name, feature_group_created_date, input_year, input_month, input_day, input_hour):

    pod_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, filter_column_value ='Pod')
    err, pod_df = pod_data.read()
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
        final_pod_df.show(truncate=False)
        
        # Running pods Only
        #final_pod_df = final_pod_df.filter(final_pod_df.pod_status == "Running")
        
        #Drop duplicates on Pod_ID and Timestamp and keep first
        final_pod_df = final_pod_df.dropDuplicates(['pod_id', 'Timestamp'])
        
        #Drop rows with nans 
        non_null_pod_df = final_pod_df.na.drop("all")
        
        #Null report
        null_report_df = null_report.report_generator(final_pod_df, processed_features)     
        null_report_df.show(truncate=False)
        
        return features_df, final_pod_df
    else:
        empty_df = pd.DataFrame()
        return empty_df
        


def pod_autoencoder_ad_feature_engineering(input_features_df, input_processed_df):

    model_parameters = input_features_df["model_parameters"]
    features =  feature_processor.cleanup(input_features_df["feature_name"].to_list())
    
    time_steps = model_parameters[0]["time_steps"]
    batch_size = model_parameters[0]["batch_size"]
    n_samples = batch_size * model_parameters[0]["sample_multiplier"]
    
    
    x_train = np.zeros((n_samples,time_steps,len(features)))
    
    pod_data = np.zeros((n_samples,time_steps,len(features)))
    for n in range(n_samples):
        ##pick random df, and normalize
        random_instance_df= input_node_processed_df.select("pod_id").orderBy(rand()).limit(1)
        pod_fe_df = input_node_processed_df[(input_node_processed_df["pod_id"] == random_instance_df.first()["pod_id"])][['Timestamp'] + features].select('*')
        pod_fe_df = pod_fe_df.sort("Timestamp")
        
        #scaler transformations
        assembler = VectorAssembler(inputCols=features, outputCol="vectorized_features")
        pod_fe_df = assembler.transform(pod_fe_df)
        scaler = StandardScaler(inputCol = "vectorized_features", outputCol = "scaled_features", withMean=True, withStd=True)
        pod_fe_df = scaler.fit(pod_fe_df).transform(pod_fe_df)
        pod_fe_df.show(truncate=False)
        
        #final X_train tensor
        start = random.choice(range(len(pod_fe_df)-time_steps))
        pod_data[n,:,:] = pod_fe_df[start:start+time_steps][scaled_features]

    print(pod_data)
    print(pod_data.shape)
    
    return pod_data

    

def pod_autoencoder_train_test_split(input_df):
    
    pod_train, pod_test = input_df.randomSplit(weights=[0.8,0.2], seed=200)

    return pod_train, pod_test
      
    