from devex_sdk import EKS_Connector, Spark_Utils
# from eks_ml_pipeline imp
# from eks_ml_pipeline import unionAll, rec_type_ad_feature_engineering, rec_type_list_generator, run_multithreading
# from eks_ml_pipeline import S3Utilities, cleanup, all_rectypes_train_test_split
from utilities import feature_processor, null_report, S3Utilities, run_multithreading, unionAll
from feature_engineering import rec_type_ad_preprocessing, rec_type_ad_feature_engineering, rec_type_list_generator, all_rectypes_train_test_split

import numpy as np
from functools import partial, reduce


def rec_type_fe_pipeline(rec_type, compute_type, feature_group_name, feature_version, partition_year, partition_month,
                         partition_day, partition_hour, spark_config_setup, input_bucket_name, input_folder_name, output_bucket_name):
    # building file name dynamically
    if partition_hour == -1:
        file_name = f'{partition_year}_{partition_month}_{partition_day}'
    elif partition_day == -1:
        file_name = f'{partition_year}_{partition_month}'
    else:
        file_name = f'{partition_year}_{partition_month}_{partition_day}_{partition_hour}'


    if rec_type == 'Node':
        aggregation_column = 'InstanceId'
    elif rec_type == 'Container':
        aggregation_column = 'container_name_pod_id'
    else:
        aggregation_column = 'pod_id'

    print(f'filename {file_name}')
    print(f'aggregatoin_column {aggregation_column}')
    
    # pre processing
    print('pre-processing')
    features_data, processed_data = rec_type_ad_preprocessing(rec_type, input_bucket_name, input_folder_name, feature_group_name,feature_version, partition_year,
                                                                                partition_month, partition_day,
                                                                                partition_hour,
                                                                                spark_config_setup)

    print(f'processed data column s: {processed_data.columns}')
    print('caching')
    processed_data.cache()

    parsing model parameters
    scaled_features = []
    model_parameters = features_data["model_parameters"].iloc[0]
    features = feature_processor.cleanup(features_data["feature_name"].to_list())
    time_steps = model_parameters["time_steps"]
    for feature in features:
        scaled_features = scaled_features + ["scaled_" + feature]

    # train, test split
    train_split = model_parameters["split_ratio"]
    test_split = round(1 - train_split, 2)
    train_data, test_data = all_rectypes_train_test_split(processed_data,
                                                          [train_split, test_split])

    # initializing s3 utils
    s3_utils = S3Utilities(output_bucket_name, feature_group_name, feature_version)

    # writing dfs to s3 bucket
    print('saving pre-processed to s3')
    # s3_utils.pyspark_write_parquet(node_processed_data, "raw_data/sparkdf", "parquet")
    #processed_data.write.mode("overwrite").parquet(f's3a://{output_bucket_name}/{feature_group_name}/{feature_version}/data/spark_df/raw_data_{file_name}/')
    s3_utils.pyspark_write_parquet(processed_data, 'data/spark_df/', 'raw_data_from_run/')

    print('saving features data to s3')
    s3_utils.awswrangler_pandas_dataframe_to_s3(features_data, "data", "pandas_df",
                                                f'raw_features_rom_run.parquet')
    print('saving raw train to s3')
    #train_data.write.mode("overwrite").parquet(f's3a://{output_bucket_name}/{feature_group_name}/{feature_version}/data/spark_df/raw_training_data_{file_name}/')
    s3_utils.pyspark_write_parquet(train_data, 'data/spark_df/', f'raw_training_data_from_run/')
    
    print('saving raw test to s3')
    # s3_utils.pyspark_write_parquet(node_test_data, "raw_testing_data/sparkdf", "parquet")
    # test_data.coalesce(10).write.mode("overwrite").parquet(f's3a://{output_bucket_name}/{feature_group_name}/{feature_version}/data/spark_df/raw_testing_data_{file_name}/')
    s3_utils.pyspark_write_parquet(train_data, 'data/spark_df/', f'raw_testing_data_from_run/')

    un-persisting processed data
    processed_data.unpersist()

    ###### Node list generator and feature engineering in same step
    
    # Create a spark session to read files from s3
    spark = Spark_Utils().get_spark()

    train_data = spark.read.parquet(f's3a://{output_bucket_name}/{feature_group_name}/{feature_version}/data/spark_df/raw_training_data_from_run/')
    print(f'train data shape: {train_data.columns}')

    test_data = spark.read.parquet(f's3a://{output_bucket_name}/{feature_group_name}/{feature_version}/data/spark_df/raw_testing_data_from_run/')
    print(f'train data shape: {test_data.columns}')

    features_data = s3_utils.read_parquet_to_pandas_df("data", "pandas_df", f'raw_features_{file_name}.parquet')


    print('generating random selected list of container ids')
    selected_train_list, processed_train_data = rec_type_list_generator('train', [train_split, test_split], train_data, aggregation_column, features_data)

    print(f'container list length: {len(selected_train_list)}')
    print('Caching')
    processed_train_data.cache()

    print('feature engineering')
    training_df_list, tensor_list = run_multithreading(rec_type_ad_feature_engineering,
                                                                           input_df=processed_train_data,
                                                                           aggregation_column=aggregation_column,
                                                                           input_features=features,
                                                                           input_scaled_features=scaled_features,
                                                                           input_time_steps=time_steps, spark=spark,
                                                                           selected_rec_type_list=selected_train_list)
    
    print(f'length of train df list: {len(training_df_list)}')
    print(f'lenght of tensir list: {len(tensor_list)}')

#     print('reshaping tensors')
#     train_tensor = np.zeros((n_samples, time_steps, len(features)))
#     for n in range(n_samples):
#         train_tensor[n, :, :] = tensor_list[n]

#     print('writing tensors rto s3')
#     s3_utils.write_tensor(train_tensor, "data", "tensors", f'training_{file_name}.npy')

#     print('Concatenating the train df lists')
#     training_df = unionAll(*training_df_list)

#     print(f'final training df columns : {training_df.columns}')

#     print('Writing train df to s3')
#     training_df.coalesce(20).write.mode("overwrite").parquet(
#         's3://emr-serverless-output-pd/emr_serverless_demo/container_autoencoder_ad/v0'
#         '.0.1/data/spark_df/training_data/')
    print('Un-persisting')
    train_data.unpersist()
    print("training and testing data_builder completed successfully")









