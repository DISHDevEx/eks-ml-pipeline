"""
these are the parameters for inference pipeline
"""
import os
from dotenv import load_dotenv

load_dotenv()

def node_inference_input():

    """
    outputs
    -------
            list of parameters for node rec type
            required for inference pipeline

    """

    rec_type = 'Node'

    ##feature engineering specs
    sampling_column = "InstanceId"

    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "9"
    partition_day = "11"
    partition_hour = "10"
    spark_config_setup = "384gb"

    ##s3 bucket for raw inference data
    data_bucketname = os.environ.get("BUCKET_NAME_OUTPUT")
    bucket_name_raw_data = os.environ.get("BUCKET_NAME_RAW_DATA")
    folder_name_raw_data = os.environ.get("FOLDER_NAME_RAW_DATA")

    return [rec_type, sampling_column,
            partition_year, partition_month,
            partition_day, partition_hour,
            spark_config_setup, data_bucketname,
            bucket_name_raw_data,folder_name_raw_data]

def pod_inference_input():

    """
    outputs
    -------
            list of parameters for pod rec type
            required for inference pipeline

    """

    rec_type = 'Pod'

    ##feature engineering specs
    sampling_column = ("kubernetes", "pod_id")

    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "7"
    partition_day = "10"
    partition_hour = "20"
    spark_config_setup = "384gb"

    ##s3 bucket for raw inference data
    data_bucketname = os.environ.get("BUCKET_NAME_OUTPUT")
    bucket_name_raw_data = os.environ.get("BUCKET_NAME_RAW_DATA")
    folder_name_raw_data = os.environ.get("FOLDER_NAME_RAW_DATA")

    return [rec_type, sampling_column,
            partition_year, partition_month,
            partition_day, partition_hour,
            spark_config_setup, data_bucketname,
            bucket_name_raw_data,folder_name_raw_data]


def container_inference_input():

    """
    outputs
    -------
            list of parameters for container rec type
            required for inference pipeline

    """

    rec_type = 'Container'

    ##feature engineering specs
    sampling_column = "InstanceId"

    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "8"
    partition_day = "20"
    partition_hour = "9"
    spark_config_setup = "384gb"

    ##s3 bucket for raw inference data
    data_bucketname = os.environ.get("BUCKET_NAME_OUTPUT")
    bucket_name_raw_data = os.environ.get("BUCKET_NAME_RAW_DATA")
    folder_name_raw_data = os.environ.get("FOLDER_NAME_RAW_DATA")

    return [rec_type, sampling_column,
            partition_year, partition_month,
            partition_day, partition_hour,
            spark_config_setup, data_bucketname,
            bucket_name_raw_data,folder_name_raw_data]
