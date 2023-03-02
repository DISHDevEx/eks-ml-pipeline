import pandas as pd
from pyspark.sql.functions import col, count, rand, row_number, get_json_object, concat_ws
from ..utilities import feature_processor
from devex_sdk import EKS_Connector, get_features

"""
Data processing function as first step of our feature engineering functions to run batch jobs that builds training 
data for Anomaly Detection models 
"""


def rec_type_ad_preprocessing(rec_type, input_feature_group_name, input_feature_group_version, input_year, input_month,
                              input_day, input_hour, input_setup, bucket_name_raw_data, folder_name_raw_data):
    """
    inputs
    ------
            rect_type: STRING
            eks record type to run data processing for

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

            bucket_name_raw_data: STRING
            s3 bucket to read raw data from

            folder_name_raw_data: STRING
            input data s3 folder name

    outputs
    -------
            features_df : processed features dataFrame (pandas df)
            processed_node_df: pre-processed dataframe (pyspark df)

    """

    pyspark_data = EKS_Connector(bucket_name=bucket_name_raw_data, folder_name=folder_name_raw_data, year=input_year,
                                 month=input_month, day=input_day, hour=input_hour, filter_column_value=rec_type,
                                 setup=input_setup)
    err, pyspark_df = pyspark_data.read()

    if err == 'PASS':

        # get features
        features_df = get_features(input_feature_group_name, input_feature_group_version)
        features = features_df["feature_name"].to_list()
        processed_features = feature_processor.cleanup(features)
        model_parameters = features_df["model_parameters"].iloc[0]
        time_steps = model_parameters["time_steps"]

        # filter initial rec_type df based on request features
        if rec_type == 'Node':
            rec_type_df = pyspark_df.select("Timestamp", "InstanceId", *processed_features)
        elif rec_type == 'Container':
            rec_type_df = pyspark_df.select("Timestamp",
                                            concat_ws("-", get_json_object(col("kubernetes"), "$.container_name"),
                                                      get_json_object(col("kubernetes"), "$.pod_id")).alias(
                                                "container_name_pod_id"), *processed_features)
        elif rec_type == 'Pod':
            rec_type_df = pyspark_df.select("Timestamp", get_json_object(col("kubernetes"), "$.pod_id").alias("pod_id"),
                                            "pod_status", *processed_features)

        rec_type_df = rec_type_df.withColumn("Timestamp", (col("Timestamp") / 1000).cast("timestamp"))

        # Drop NA
        cleaned_rec_type_df = rec_type_df.na.drop(subset=processed_features)

        # Quality(timestamp filtered) rec_types
        if rec_type == 'Node':
            quality_filtered_rec_type_df = cleaned_rec_type_df.groupBy("InstanceId").agg(
                count("Timestamp").alias("timestamp_count"))
        elif rec_type == 'Container':
            quality_filtered_rec_type_df = cleaned_rec_type_df.groupBy("container_name_pod_id").agg(
                count("Timestamp").alias("timestamp_count"))
        elif rec_type == 'Pod':
            cleaned_rec_type_df = cleaned_rec_type_df.filter(col("pod_status") == "Running")
            quality_filtered_rec_type_df = cleaned_rec_type_df.groupBy("pod_id").agg(
                count("Timestamp").alias("timestamp_count"))

        # to get data that is closer to 1min apart
        quality_filtered_rec_types = quality_filtered_rec_type_df.filter(col("timestamp_count") >= 2 * time_steps)

        # Processed Node DF
        if rec_type == 'Node':
            processed_rec_type_df = cleaned_rec_type_df.filter(
                col("InstanceId").isin(quality_filtered_rec_types["InstanceId"]))
        elif rec_type == 'Container':
            processed_rec_type_df = cleaned_rec_type_df.filter(
                col("container_name_pod_id").isin(quality_filtered_rec_types["container_name_pod_id"]))
        elif rec_type == 'Pod':
            processed_rec_type_df = cleaned_rec_type_df.filter(col("pod_id").isin(quality_filtered_rec_types["pod_id"]))
            processed_rec_type_df = processed_rec_type_df.sort("Timestamp")
            # Drop duplicates on Pod_ID and Timestamp and keep first
            processed_rec_type_df = processed_rec_type_df.dropDuplicates(['pod_id', 'Timestamp'])
            # Drop rows with nans
            processed_rec_type_df = processed_rec_type_df.na.drop("all")

        return features_df, processed_rec_type_df

    else:
        empty_df = pd.DataFrame()
        return empty_df, empty_df