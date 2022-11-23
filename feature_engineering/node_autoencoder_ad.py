from msspackages import Pyspark_data_ingestion, get_features
from pyspark.sql.functions import col


def node_autoencoder_ad_preprocessing(feature_group_name, feature_group_created_date, input_year, input_month, input_day, input_hour):

    node_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, filter_column_value ='Node')
    node_spark_context= node_data.get_spark()
    err, node_df = node_data.read()
    if err == 'PASS':
        features_df = get_features(feature_group_name,feature_group_created_date)
        features = features_df["feature_name"].to_list()
        print(features)
        node_df = node_df. withColumn("Timestamp",(col("Timestamp")/1000).cast("timestamp")).select("Timestamp","InstanceId","node_cpu_utilization","node_memory_utilization","node_network_total_bytes")
        node_df = node_df.na.drop(subset=["node_cpu_utilization","node_memory_utilization","node_network_total_bytes"])
        return node_df
    else:
        empty_df = pd.DataFrame()
        return empty_df
    
    

def node_autoencoder_ad_feature_engineering(input_processed_df):
    
    featured_engineered_df  = input_processed_df
    return featured_engineered_df




def write_to_s3(input_featured_engineered_df):
    return 0
    
    