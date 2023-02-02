import pandas as pd
from pyspark.sql.functions import col, count, rand, row_number
from ..utilities import feature_processor
from devex_sdk import EKS_Connector, get_features

"""
this feature engineering functions will help us run bach jobs that builds training data for Anomaly Detection models
"""


def node_ad_preprocessing(input_feature_group_name, input_feature_group_version, input_year, input_month, input_day,
                          input_hour, input_setup="default"):
    """
    inputs
    ------
            input_feature_group_name: STRING
            json name to get the required features

            input_feature_group_version: STRING
            json version to get the latest features

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
            features_df : processed features dataFrame (pandas df)
            processed_node_df: pre processed node dataframe (pyspark df)
            
    """

    pyspark_node_data = EKS_Connector(year = input_year, month = input_month, day = input_day, hour = input_hour, setup = input_setup, filter_column_value ='Node')
    err, pyspark_node_df = pyspark_node_data.read()

    if err == 'PASS':

        # get features
        features_df = get_features(input_feature_group_name, input_feature_group_version)
        features = features_df["feature_name"].to_list()
        processed_features = feature_processor.cleanup(features)

        model_parameters = features_df["model_parameters"].iloc[0]
        time_steps = model_parameters["time_steps"]

        # filter inital node df based on request features
        node_df = pyspark_node_df.select("Timestamp", "InstanceId", *processed_features)
        node_df = node_df.withColumn("Timestamp", (col("Timestamp") / 1000).cast("timestamp"))

        # Drop NA
        cleaned_node_df = node_df.na.drop(subset=processed_features)

        # Quality(timestamp filtered) nodes
        quality_filtered_node_df = cleaned_node_df.groupBy("InstanceId").agg(
            count("Timestamp").alias("timestamp_count"))
        # to get data that is closer to 1min apart
        quality_filtered_nodes = quality_filtered_node_df.filter(col("timestamp_count") >= 2 * time_steps)

        # Processed Node DF
        processed_node_df = cleaned_node_df.filter(col("InstanceId").isin(quality_filtered_nodes["InstanceId"]))

        return features_df, processed_node_df

    else:
        empty_df = pd.DataFrame()
        return empty_df, empty_df
