from devex_sdk import EKS_Connector, Spark_Utils
from eks_ml_pipeline.utilities import feature_processor, null_report, S3Utilities, run_multithreading, unionAll
from eks_ml_pipeline import rec_type_ad_preprocessing, rec_type_ad_feature_engineering, rec_type_list_generator, \
    all_rectypes_train_test_split
from eks_ml_pipeline import EMRServerless

import numpy as np
from functools import partial, reduce


class FeatureEngineeringPipeline:
    """
    Allows running of data processing and feature engineering steps

    """

    def __init__(self, feature_engineering_inputs, aggregation_column, rec_type: str = None, compute_type: str = None):

        self.feature_group_name = feature_engineering_inputs[0]
        self.feature_version = feature_engineering_inputs[1]
        self.partition_year = feature_engineering_inputs[2]
        self.partition_month = feature_engineering_inputs[3]
        self.partition_day = feature_engineering_inputs[4]
        self.partition_hour = feature_engineering_inputs[5]
        self.spark_config_setup = feature_engineering_inputs[6]
        self.bucket = feature_engineering_inputs[7]
        self.bucket_name_raw_data = feature_engineering_inputs[8]
        self.folder_name_raw_data = feature_engineering_inputs[9]
        self.rec_type = rec_type
        self.compute_type = compute_type
        self.aggregation_column = aggregation_column

    def run_preproceesing(self):

        print(self.bucket_name_raw_data)
        print(self.folder_name_raw_data)

        features_data, processed_data = rec_type_ad_preprocessing(rec_type=self.rec_type,
                                                                  input_bucket_name=self.bucket_name_raw_data,
                                                                  input_folder_name=self.folder_name_raw_data,
                                                                  input_feature_group_name=self.feature_group_name,
                                                                  input_feature_group_version=self.feature_version,
                                                                  input_year=self.partition_year,
                                                                  input_month=self.partition_month,
                                                                  input_day=self.partition_day,
                                                                  input_hour=self.partition_hour,
                                                                  input_setup=self.spark_config_setup)

        print(f'processed data columns: {processed_data.columns}')
        print('caching')
        processed_data.cache()

        # parsing model parameters
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
        s3_utils = S3Utilities(self.bucket, self.feature_group_name, self.feature_version)

        # writing dfs to s3 bucket
        print('saving pre-processed to s3')
        # processed_data.write.mode("overwrite").parquet(f's3a://{output_bucket_name}/{feature_group_name}/{feature_version}/data/spark_df/raw_data_{file_name}/')
        s3_utils.pyspark_write_parquet(processed_data, 'data/spark_df/', 'raw_data_from_run/')

        print('saving features data to s3')
        s3_utils.awswrangler_pandas_dataframe_to_s3(features_data, "data", "pandas_df",
                                                    f'raw_features_rom_run.parquet')
        print('saving raw train to s3')
        # train_data.write.mode("overwrite").parquet(f's3a://{output_bucket_name}/{feature_group_name}/{feature_version}/data/spark_df/raw_training_data_{file_name}/')
        s3_utils.pyspark_write_parquet(train_data, 'data/spark_df/', f'raw_training_data_from_run/')

        print('saving raw test to s3')
        # test_data.coalesce(10).write.mode("overwrite").parquet(f's3a://{output_bucket_name}/{feature_group_name}/{feature_version}/data/spark_df/raw_testing_data_{file_name}/')
        s3_utils.pyspark_write_parquet(train_data, 'data/spark_df/', f'raw_testing_data_from_run/')

        # un-persisting processed data
        processed_data.unpersist()

    def run_feature_engineering(self):
        # Create a spark session to read files from s3
        spark = Spark_Utils().get_spark()

        train_data = spark.read.parquet(
            f's3a://{self.bucket}/{self.feature_group_name}/{self.feature_version}/data/spark_df/raw_training_data_from_run/')
        print(f'train data shape: {train_data.columns}')

        test_data = spark.read.parquet(
            f's3a://{self.bucket}/{self.feature_group_name}/{self.feature_version}/data/spark_df/raw_testing_data_from_run/')
        print(f'train data shape: {test_data.columns}')

        s3_utils = S3Utilities(self.bucket, self.feature_group_name, self.feature_version)

        #features_data = s3_utils.read_parquet_to_pandas_df("data", "pandas_df", f'raw_features_{file_name}.parquet')
        features_data = s3_utils.read_parquet_to_pandas_df("data", "pandas_df", 'raw_features_rom_run.parquet')

        # parsing model parameters
        scaled_features = []
        model_parameters = features_data["model_parameters"].iloc[0]
        features = feature_processor.cleanup(features_data["feature_name"].to_list())
        time_steps = model_parameters["time_steps"]
        for feature in features:
            scaled_features = scaled_features + ["scaled_" + feature]

        # train, test split
        train_split = model_parameters["split_ratio"]
        test_split = round(1 - train_split, 2)

        print('generating random selected list of container ids')
        selected_train_list, processed_train_data = rec_type_list_generator('train', [train_split, test_split],
                                                                            train_data, self.aggregation_column,
                                                                            features_data)

        print(f'container list length: {len(selected_train_list)}')
        print('Caching')
        processed_train_data.cache()

        print('feature engineering')
        training_df_list, tensor_list = run_multithreading(rec_type_ad_feature_engineering,
                                                           input_df=processed_train_data,
                                                           aggregation_column=self.aggregation_column,
                                                           input_features=features,
                                                           input_scaled_features=scaled_features,
                                                           input_time_steps=time_steps, spark=spark,
                                                           selected_rec_type_list=selected_train_list)

        print(f'length of train df list: {len(training_df_list)}')
        print(f'lenght of tensir list: {len(tensor_list)}')

        print('reshaping tensors')
        train_tensor = np.zeros((n_samples, time_steps, len(features)))
        for n in range(n_samples):
            train_tensor[n, :, :] = tensor_list[n]

        print('writing tensors rto s3')
        s3_utils.write_tensor(train_tensor, "data", "tensors", f'training_{file_name}.npy')

        print('Concatenating the train df lists')
        training_df = unionAll(*training_df_list)

        print(f'final training df columns : {training_df.columns}')

        print('Writing train df to s3')
        training_df.coalesce(20).write.mode("overwrite").parquet(
            's3://emr-serverless-output-pd/emr_serverless_demo/container_autoencoder_ad/v0'
            '.0.1/data/spark_df/training_data/')
        print('Un-persisting')
        train_data.unpersist()
        print("training and testing data_builder completed successfully")


    def run_in_sagemaker(self):
        self.run_preproceesing()
        self.run_feature_engineering()
        
    def run_in_emr(self):
        application_id = '00f6mv29kbd4e10l'
        s3_bucket_name = self.bucket
        zipped_env_path = 's3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/spark_dependency/pyspark_deps_github.tar.gz'
        #emr_entry_point = '/home/sagemaker-user/github/eks-ml-pipeline/eks_ml_pipeline/emr_job.py'
        #emr_entry_point = 's3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/emr_entry_point/emr_job.py'
        #emr_entry_point = 's3://dish-5g.core.pd.g.dp.eks.logs.e/emr_serverless/code/spark_dependency/pyspark_deps_github.tar.gz'
        emr_entry_point = 'local:///usr/pyspark_deps_github/lib/python3.7/site-packages/eks_ml_pipeline/emr_job.py'
        
        #zipped_env_path = f's3://{s3_bucket_name}/emr_serverless/code/spark_dependency/pyspark_deps_github.tar.gz'
        
        # if rec_type == 'Node':
        #     emr_entry_point = 
        # elif rec_type == 'Pod':
        #     emr_entry_point = 
        # elif rec_type == 'Container':
        #     emr_entry_point = 
        
        
        emr_serverless = EMRServerless()
        print("Starting EMR Serverless Spark App")
        # Start the application; skips this step automatically if the application is already in 'Started' state
        emr_serverless.start_application(application_id)
        print(emr_serverless)
        
        
        # Run (and wait for) a Spark job
        print("Submitting new Spark job")
        job_run_id = emr_serverless.run_spark_job(
            script_location=emr_entry_point,
            #job_role_arn=serverless_job_role_arn,
            application_id = application_id,
            #arguments=[f"s3://{s3_bucket_name}/emr_serverless/output"],
            s3_bucket_name=s3_bucket_name,
            zipped_env_path = zipped_env_path
        )
        
        emr_serverless.fetch_driver_log(s3_bucket_name)


#     if compute_type = 'sagemaker':
#         run_preproceesing()
#         run_feature_engineering()
#     elif compute_type = 'emr_serverless':
    
#         run_in_emr(run_preproceesing())
#         run_in_emr(run_feature_engineering())
