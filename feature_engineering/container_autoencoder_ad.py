import numpy as np
import random
from utilities import feature_processor, null_report
from msspackages import Pyspark_data_ingestion, get_features
from pyspark.sql.functions import col, count, rand
from sklearn.preprocessing import StandardScaler


def container_autoencoder_ad_preprocessing(feature_group_name, feature_group_created_date, input_year, input_month, input_day, input_hour):

    node_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, filter_column_value ='Container')
    err, node_df = node_data.read()

    if err == 'PASS':

        #get features
        features_df = get_features(feature_group_name,feature_group_created_date)
        features = features_df["feature_name"].to_list()
        processed_features = feature_processor.cleanup(features)
    
        #filter inital node df based on request features
        container_df = container_df.select("Timestamp", "kubernetes", *processed_features)
        container_df = container_df.withColumn("Timestamp",(col("Timestamp")/1000).cast("timestamp"))
        
        # Drop NA
        cleaned_container_df = container_df.na.drop(subset=processed_features)
        
        cleaned_container_df.show(truncate=False)

        
        #Quality(timestamp filtered) nodes
        quality_filtered_container_df = cleaned_container_df.groupBy("InstanceId").agg(count("Timestamp").alias("timestamp_count"))
        quality_filtered_containers = quality_filtered_container_df.filter(col("timestamp_count").between(45,75))
        

        #Processed Node DF                                                      
        processed_container_df = cleaned_container_df.filter(col("InstanceId").isin(quality_filtered_nodes["InstanceId"]))
        
        #Null report
        null_report_df = null_report.report_generator(processed_container_df, processed_features)     
        #null_report_df.show(truncate=False)
        
        return features_df, processed_container_df

    else:
        empty_df = pd.DataFrame()
        return empty_df, empty_df
    
    

def container_autoencoder_ad_feature_engineering(input_features_df, input_processed_df):

    model_parameters = input_features_df["model_parameters"]
    features =  feature_processor.cleanup(input_features_df["feature_name"].to_list())
    
    time_steps = model_parameters[0]["time_steps"]
    batch_size = model_parameters[0]["batch_size"]
    n_samples = batch_size * model_parameters[0]["sample_multiplier"]
    
    
    x_train = np.zeros((n_samples,time_steps,len(features)))
    
    for b in range(600):

        ##pick random df, and normalize
        random_instance_id = input_processed_df.select("InstanceId").orderBy(rand()).limit(1)
        filtered_instance_id = (input_processed_df["InstanceId"]==random_instance_id["InstanceId"][0])
        train_df = input_processed_df[filtered_instance_id][['Timestamp'] + features].select('*')
        train_df.show(truncate=False)
        train_df = train_df.sort_values(by='Timestamp').reset_index(drop=True)
        
        #scaler transformations
        train_df[features] = StandardScaler().fit_transform(train_df[features])
        
        
        start = random.choice(range(len(train_df)-time_steps))
        x_train[b,:,:] = train_df[start:start+time_steps][features]

    final_training_data  = x_train
    print(final_training_data)
    
    return final_training_data

    
    