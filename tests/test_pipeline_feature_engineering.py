# """
# Contain all tests pertaining to pipelines that train models.
# """
# from eks_ml_pipeline import FeatureEngineeringPipeline, rec_type_ad_preprocessing, rec_type_list_generator, \
#     rec_type_ad_feature_engineering, all_rectypes_train_test_split, S3Utilities
# from devex_sdk import EKS_Connector
# import boto3
# import pytest


# def test_rec_type_ad_preprocessing(Spark, Spark_context, bucket_name):
#     """
#     Verify the pre-processing logic for the autoencoder feature engineering pipeline.
#     Checks if the raw data is read and pre-processed.

#     Inputs: fixtures of spark and bucket name.
#     Output: None
#     """
#     folder_name = 'pytest_autoencoder_ad/v0.0.1/data/spark_df/EKS_SAMPLE_DATA.snappy.parquet'
#     feature_group_name = 'pytest_autoencoder_ad'
#     feature_group_version = 'v0.0.1'
#     rec_type = 'Node'

#     pytest_obj = EKS_Connector(bucket_name, folder_name, filter_column_value=rec_type)
#     err_code, input_raw_df = pytest_obj.read()

#     features_df, processed_rec_type_df = rec_type_ad_preprocessing(input_raw_df, feature_group_name,
#                                                                    feature_group_version, rec_type)

#     assert len(features_df.columns) == 11
#     assert len(processed_rec_type_df.columns) == 5


# def test_rec_type_list_generator(Spark, Spark_context, bucket_name):
#     file_name_features_df = 'sample_raw_features_2022_9_29_1'
#     folder_name_raw_df = 'pytest_autoencoder_ad/v0.0.1/data/spark_df/raw_training_data_2022_9_29_1/'
#     feature_group_name = 'pytest_autoencoder_ad'
#     feature_group_version = 'v0.0.1'
#     input_data_type = 'train'
#     aggregation_column = 'InstanceId'

#     s3_util = S3Utilities(
#         bucket_name=bucket_name,
#         model_name=feature_group_name,
#         version=feature_group_version,
#     )

#     features_df = s3_util.read_parquet_to_pandas_df("data", "pandas_df", f'{file_name_features_df}.parquet')
#     input_preprocessed_df = Spark.read.parquet(f's3a://{bucket_name}/{folder_name_raw_df}')

#     model_parameters = features_df["model_parameters"].iloc[0]

#     # train, test split
#     train_split = model_parameters["split_ratio"]
#     test_split = round(1 - train_split, 2)

#     selected_rec_type_list, processed_rec_type_data = rec_type_list_generator(input_data_type,
#                                                                               [train_split, test_split],
#                                                                               input_preprocessed_df, aggregation_column,
#                                                                               features_df)

#     assert len(processed_rec_type_data.columns) == 6
#     assert len(selected_rec_type_list) == 360