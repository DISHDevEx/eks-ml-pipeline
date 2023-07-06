from devex_sdk import EKS_Connector, Spark_Utils
from eks_ml_pipeline.utilities import feature_processor, null_report, S3Utilities, run_multithreading, unionAll
from eks_ml_pipeline import rec_type_ad_preprocessing, rec_type_ad_feature_engineering, rec_type_list_generator, \
    all_rectypes_train_test_split, read_raw_input_data
from pyspark.sql import SparkSession  # new addition
from eks_ml_pipeline import EMRServerless

import numpy as np
from functools import partial, reduce


class FeatureEngineeringPipeline:
    """
    Allows running of data processing and feature engineering steps.
    """

    def __init__(self, feature_engineering_inputs, rec_type: str = None, compute_type: str = None,
                 input_data_type: str = None):

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
        self.input_data_type = input_data_type

        self.aggregation_column = None
        self.s3_utilities = None
        self.file_name = None
        self.create_file_path()
        self.initialize_s3()

    def initialize_s3(self):
        """Initialize s3 utilities class."""

        self.s3_utilities = S3Utilities(
            bucket_name=self.bucket,
            model_name=self.feature_group_name,
            version=self.feature_version,
        )

    def create_file_path(self):
        """Create File Paths."""

        if self.partition_hour == -1:
            self.file_name = f'{self.partition_year}_{self.partition_month}_{self.partition_day}'
        elif self.partition_day == -1:
            self.file_name = f'{self.partition_year}_{self.partition_month}'
        else:
            self.file_name = f'{self.partition_year}_{self.partition_month}_{self.partition_day}_{self.partition_hour}'
    
    def run_preprocessing(self):
        """Run data pre-processing step."""

        # Create a spark session to read files from s3
        spark = Spark_Utils().get_spark()
        
        raw_input_df = read_raw_input_data(rec_type=self.rec_type,
                                           input_year=self.partition_year,
                                           input_month=self.partition_month,
                                           input_day=self.partition_day,
                                           input_hour=self.partition_hour,
                                           input_setup=self.spark_config_setup,
                                           bucket_name_raw_data=self.bucket_name_raw_data,
                                           folder_name_raw_data=self.folder_name_raw_data)
        
        
        features_data, processed_data = rec_type_ad_preprocessing(input_df=raw_input_df, 
                                                                  rec_type = self.rec_type, 
                                                                  input_feature_group_name = self.feature_group_name, 
                                                                  input_feature_group_version = self.feature_version)

        # caching
        #processed_data.persist()

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

        # writing dfs to s3 bucket
        self.s3_utilities.pyspark_write_parquet(processed_data, 'data/spark_df', f'raw_data_{self.file_name}')

        self.s3_utilities.awswrangler_pandas_dataframe_to_s3(features_data, "data", "pandas_df",
                                                             f'raw_features_{self.file_name}.parquet')
        
        self.s3_utilities.pyspark_write_parquet(train_data, 'data/spark_df', f'raw_training_data_{self.file_name}')

        self.s3_utilities.pyspark_write_parquet(test_data, 'data/spark_df', f'raw_testing_data_{self.file_name}')

        # un-persisting processed data
        #processed_data.unpersist()

    def run_feature_engineering(self):
        """Run feature engineering step"""

        # Create a spark session to read files from s3
        spark = Spark_Utils().get_spark()

        if self.rec_type == 'Node':
            self.aggregation_column = 'InstanceId'
        elif self.rec_type == 'Container':
            self.aggregation_column = 'container_name_pod_id'
        else:
            self.aggregation_column = 'pod_id'

        # read train data
        train_data = spark.read.parquet(
            f's3a://{self.bucket}/{self.feature_group_name}/{self.feature_version}/data/spark_df/raw_training_data_{self.file_name}/')

        # read test data
        test_data = spark.read.parquet(
            f's3a://{self.bucket}/{self.feature_group_name}/{self.feature_version}/data/spark_df/raw_testing_data_{self.file_name}/')

        features_data = self.s3_utilities.read_parquet_to_pandas_df("data/", "pandas_df",
                                                                    f'raw_features_{self.file_name}.parquet')

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

        print('generating random selected list of ids')
        if self.input_data_type == 'train':
            selected_rec_type_list, processed_rec_type_data = rec_type_list_generator('train',
                                                                                      [train_split, test_split],
                                                                                      train_data,
                                                                                      self.aggregation_column,
                                                                                      features_data)
        elif self.input_data_type == 'test':
            selected_rec_type_list, processed_rec_type_data = rec_type_list_generator('test', [train_split, test_split],
                                                                                      test_data,
                                                                                      self.aggregation_column,
                                                                                      features_data)

        # Caching
        processed_rec_type_data.cache()

        # feature engineering
        print('multithreading')
        rec_type_df_list, tensor_list = run_multithreading(rec_type_ad_feature_engineering,
                                                           input_df=processed_rec_type_data,
                                                           aggregation_column=self.aggregation_column,
                                                           input_features=features,
                                                           input_scaled_features=scaled_features,
                                                           input_time_steps=time_steps, spark=spark,
                                                           selected_rec_type_list=selected_rec_type_list)

        # reshaping tensors
        train_tensor = np.array(tensor_list)
        
        # writing artifacts to s3
        self.s3_utilities.write_tensor(train_tensor, "data", "tensors",
                                       f'{self.input_data_type}ing_{self.file_name}.npy')

        # concatenating the list of dfs
        rec_type_df = unionAll(*rec_type_df_list)

        # writing df to s3'
        self.s3_utilities.pyspark_write_parquet(rec_type_df, 'data/spark_df',
                                                f'{self.input_data_type}ing_data_{self.file_name}')
        # unpersist
        processed_rec_type_data.unpersist()
        
        print(f"{self.input_data_type} data building completed successfully")

    def run_in_sagemaker(self):
        """Run pre-processing and feature engineering steps in SageMaker."""

        print('running data pre-processing step')
        self.run_preprocessing()
        print('running data engineering step')
        self.run_feature_engineering()

    def run_in_emr(self, job_type='processing'):
        """Run pre-processing and feature engineering steps in EMR Serverless."""
        application_id = '00f6muv5dgv8n509'
        s3_bucket_name = self.bucket
        zipped_env_path = f's3://{s3_bucket_name}/emr_serverless/code/spark_dependency/pyspark_deps_github.tar.gz'

        if self.rec_type == 'Node':
            emr_entry_point_processing = f's3://{s3_bucket_name}/emr_serverless/code/emr_entry_point/autoencoder_processing_node.py'
            if self.input_data_type == 'train':
                emr_entry_point_feature_engineering = f's3://{s3_bucket_name}/emr_serverless/code/emr_entry_point/autoencoder_fe_node_train.py'
            else:
                emr_entry_point_feature_engineering = f's3://{s3_bucket_name}/emr_serverless/code/emr_entry_point/autoencoder_fe_node_test.py'
        elif self.rec_type == 'Pod':
            emr_entry_point_processing = f's3://{s3_bucket_name}/emr_serverless/code/emr_entry_point/autoencoder_processing_pod.py'
            if self.input_data_type == 'train':
                emr_entry_point_feature_engineering = f's3://{s3_bucket_name}/emr_serverless/code/emr_entry_point/autoencoder_fe_pod_train.py'
            else:
                emr_entry_point_feature_engineering = f's3://{s3_bucket_name}/emr_serverless/code/emr_entry_point/autoencoder_fe_pod_test.py'
        elif self.rec_type == 'Container':
            emr_entry_point_processing = f's3://{s3_bucket_name}/emr_serverless/code/emr_entry_point/autoencoder_processing_container.py'
            if self.input_data_type == 'train':
                emr_entry_point_feature_engineering = f's3://{s3_bucket_name}/emr_serverless/code/emr_entry_point/autoencoder_fe_container_train.py'
            else:
                emr_entry_point_feature_engineering = f's3://{s3_bucket_name}/emr_serverless/code/emr_entry_point/autoencoder_fe_container_test.py'

        emr_serverless = EMRServerless()
        print("Starting EMR Serverless Spark App")
        # Start the application; skips this step automatically if the application is already in 'Started' state
        emr_serverless.start_application(application_id)
        print(emr_serverless)

        # Run (and wait for) a Spark job
        print("Submitting new Spark job")

        if job_type == 'processing':
            job_run_id = emr_serverless.run_spark_job(
                script_location=emr_entry_point_processing,
                application_id=application_id,
                s3_bucket_name=s3_bucket_name,
                zipped_env_path=zipped_env_path
            )
        elif job_type == 'feature_engineering':
            job_run_id = emr_serverless.run_spark_job(
                script_location=emr_entry_point_feature_engineering,
                application_id=application_id,
                s3_bucket_name=s3_bucket_name,
                zipped_env_path=zipped_env_path
            )

        emr_serverless.fetch_driver_log(s3_bucket_name)