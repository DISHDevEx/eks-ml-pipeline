import numpy as np
import random
from utilities import feature_processor, null_report
from msspackages import Pyspark_data_ingestion, get_features
from pyspark.sql.functions import col, count, rand
from sklearn.preprocessing import StandardScaler


# def pod_autoencoder_ad_preprocessing(feature_group_name, feature_group_created_date, input_year, input_month, input_day, input_hour):

#     node_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, filter_column_value ='Pod')
#     err, pod_df = node_data.read()

#     if err == 'PASS':

#         #get features
#         features_df = get_features(feature_group_name,feature_group_created_date)
#         features = features_df["feature_name"].to_list()
#         processed_features = feature_processor.cleanup(features)
    
#         #filter inital node df based on request features
#         pod_df = pod_df.select("Timestamp", "InstanceId", *processed_features)
#         pod_df = pod_df.withColumn("Timestamp",(col("Timestamp")/1000).cast("timestamp")) #why divide by 1000?
        
#         # Drop NA
#         cleaned_pod_df = pod_df.na.drop(subset=processed_features)

        
#         #Quality(timestamp filtered) nodes
#         quality_filtered_pod_df = cleaned_pod_df.groupBy("InstanceId").agg(count("Timestamp").alias("timestamp_count"))
#         quality_filtered_nodes = quality_filtered_pod_df.filter(col("timestamp_count").between(45,75))
        

#         #Processed Node DF                                                      
#         processed_pod_df = cleaned_pod_df.filter(col("InstanceId").isin(quality_filtered_nodes["InstanceId"]))
        
#         #Null report
#         null_report_df = null_report.report_generator(processed_pod_df, processed_features)     
#         #null_report_df.show(truncate=False)
        
#         return features_df, processed_pod_df

#     else:
#         empty_df = pd.DataFrame()
#         return empty_df, empty_df
    

    
def pod_autoencoder_ad_preprocessing(feature_group_name, feature_group_created_date, input_year, input_month, input_day, input_hour):

    pod_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, filter_column_value ='Pod')
    err, pod_df = pod_data.read()
    pod_data = pod_data.select(*pod_data.columns,get_json_object(col("kubernetes"),"$.pod_name").alias("pod_name"))


    if err == 'PASS':
        #get features
        features_df = get_features(feature_group_name,feature_group_created_date)
        features = features_df["feature_name"].to_list()
        
        processed_features = cleanup(features)
    
        #filter inital pod df based on request features
        pod_df = pod_df.select("Timestamp", *processed_features)
        pod_df = pod_df.withColumn("Timestamp",(col("Timestamp")/1000).cast("timestamp"))
        cleaned_pod_df = pod_df.na.drop(subset=processed_features)
        
        #Quality(timestamp filtered) pods
        quality_filtered_pod_df = cleaned_pod_df.groupBy("pod_id").agg(count("Timestamp").alias("timestamp_count"))
        quality_filtered_pods = quality_filtered_pod_df.filter(col("timestamp_count").nsmallest(45))
        

        #Processed pod DF                                                      
        final_pod_df = cleaned_pod_df.filter(col("pod_id").isin(quality_filtered_pods["pod_id"]))
        final_pod_df = final_pod_df.sort("Timestamp")
        final_pod_df.show(truncate=False)
        
        # Running pods Only
        
        final_pod_df = final_pod_df.filter(final_pod_df.pod_status == "Running")
        
        #Drop duplicates on Pod_ID and Timestamp and keep first
        
        final_pod_df = final_pod_df.dropDuplicates(['pod_id', 'Timestamp'])
        
        #Drop rows with nans 
        non_null_pod_df = final_pod_df.na.drop("all")
        
        #Null report
        null_report_df = report_generator(final_pod_df, processed_features)     
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
    
    for b in range(600):

        ##pick random df, and normalize
        random_instance_id = input_processed_df.select("InstanceId").orderBy(rand()).limit(1)
        random_instance_id.show(truncate=False)
        filtered_instance_id = (input_processed_df['InstanceId']==random_instance_id)
        train_df = input_processed_df[filtered_instance_id][['Timestamp'] + features].copy()
        train_df = train_df.sort_values(by='Timestamp').reset_index(drop=True)
        
        #scaler transformations
        train_df[features] = StandardScaler().fit_transform(train_df[features])
        
        
        start = random.choice(range(len(train_df)-time_steps))
        x_train[b,:,:] = train_df[start:start+time_steps][features]

    final_training_data  = x_train
    print(final_training_data)
    
    return final_training_data

    
    