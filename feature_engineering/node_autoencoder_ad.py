from msspackages import Pyspark_data_ingestion, get_features
from pyspark.sql.functions import col, count, when, isnan
from utilities import cleanup, report_generator


def node_autoencoder_ad_preprocessing(feature_group_name, feature_group_created_date, input_year, input_month, input_day, input_hour):

    node_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, filter_column_value ='Node')
    err, node_df = node_data.read()

    if err == 'PASS':
        #get features
        features_df = get_features(feature_group_name,feature_group_created_date)
        features = features_df["feature_name"].to_list()
        
        processed_features = cleanup(features)
    
        #filter inital node df based on request features
        node_df = node_df.select("Timestamp", *processed_features)
        node_df = node_df.withColumn("Timestamp",(col("Timestamp")/1000).cast("timestamp"))
        cleaned_node_df = node_df.na.drop(subset=processed_features)

        
        #Quality(timestamp filtered) nodes
        quality_filtered_node_df = cleaned_node_df.groupBy("InstanceId").agg(count("Timestamp").alias("timestamp_count"))
        quality_filtered_nodes = quality_filtered_node_df.filter(col("timestamp_count").between(45,75))
        

        #Processed Node DF                                                      
        final_node_df = cleaned_node_df.filter(col("InstanceId").isin(quality_filtered_nodes["InstanceId"]))
        final_node_df = final_node_df.sort("Timestamp")
        final_node_df.show(truncate=False)
        
        #Null report
        null_report_df = report_generator(final_node_df, processed_features)     
        null_report_df.show(truncate=False)
        
        return features_df, final_node_df
    else:
        empty_df = pd.DataFrame()
        return empty_df
    
    

def node_autoencoder_ad_feature_engineering(input_features_df, input_processed_df):

    features_df = input_features_df
    final_training_data  = input_processed_df
    
    print(features_df)
    print(final_training_data)
    return final_training_data

    
    