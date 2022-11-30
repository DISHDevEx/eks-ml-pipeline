from msspackages import Pyspark_data_ingestion, get_features
from pyspark.sql.functions import col


def container_autoencoder_ad_preprocessing(feature_group_name, feature_group_created_date, input_year, input_month, input_day, input_hour):

    container_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, filter_column_value ='Container')
    err, container_df = container_data.read()
    if err == 'PASS':
        features_df = get_features(feature_group_name,feature_group_created_date)
        features = features_df["feature_name"].to_list()
        
        return features_df, container_df
    else:
        empty_df = pd.DataFrame()
        return empty_df, features_df
    
    

def container_autoencoder_ad_feature_engineering(input_processed_df):
    
    featured_engineered_df  = input_processed_df
    return featured_engineered_df

    
    