from msspackages import Pyspark_data_ingestion
from msspackages import get_features


def node_autoencoder_ad_preprocessing():

    node_data = Pyspark_data_ingestion(year = '2022', month = '10', day = "10", hour = "10", filter_column_value ='Node')
    node_spark_context= node_data.get_spark()
    err, node_df = node_data.read()
    if err == 'PASS':
        features_df = get_features("node_autoencoder_ad_features","11-21-2022")
        return node_df, features_df
    else:
        empty_df = pd.DataFrame()
        return empty_df, empty_df