import os


def node_autoencoder_fe_input():
    """
    outputs
    -------
            list of parameters for node rec type
            required for autoencoder feature engineering
            pipeline

    """
    ##feature parameters
    feature_group_name = "node_autoencoder_ad"
    feature_version = "v0.0.2"

    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "9"
    partition_day = "29"
    partition_hour = "1"
    spark_config_setup = "384gb"

    ##s3 bucket parameters
    bucket = os.environ.get("BUCKET_NAME_OUTPUT")
    bucket_name_raw_data = os.environ.get("BUCKET_NAME_RAW_DATA")
    folder_name_raw_data = os.environ.get("FOLDER_NAME_RAW_DATA")
    return [feature_group_name, feature_version,
            partition_year, partition_month, partition_day,
            partition_hour, spark_config_setup,
            bucket,bucket_name_raw_data,folder_name_raw_data]



def node_pca_fe_input():
    """
    outputs
    -------
            list of parameters for node rec type
            required for pca feature engineering
            pipeline

    """
    ##feature parameters
    feature_group_name = "node_pca_ad"
    feature_version = "v0.0.1"

    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "9"
    partition_day = "29"
    partition_hour = "1"
    spark_config_setup = "384gb"

    ##s3 bucket parameters
    bucket = os.environ.get("BUCKET_NAME_OUTPUT")

    ##s3 bucket parameters
    bucket = os.environ.get("BUCKET_NAME_OUTPUT")
    bucket_name_raw_data = os.environ.get("BUCKET_NAME_RAW_DATA")
    folder_name_raw_data = os.environ.get("FOLDER_NAME_RAW_DATA")
    return [feature_group_name, feature_version, partition_year, partition_month, partition_day,
            partition_hour, spark_config_setup, bucket, bucket_name_raw_data, folder_name_raw_data]


def node_hmm_fe_input():
    """
    outputs
    -------
            list of parameters for node rec type
            required for hmm feature engineering
            pipeline

    """
    ##feature parameters
    feature_group_name = "node_hmm_ad"
    feature_version = "v0.0.1"

    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "9"
    partition_day = "29"
    partition_hour = "1"
    spark_config_setup = "384gb"

    ##s3 bucket parameters
    bucket = os.environ.get("BUCKET_NAME_OUTPUT")

    ##s3 bucket parameters
    bucket = os.environ.get("BUCKET_NAME_OUTPUT")
    bucket_name_raw_data = os.environ.get("BUCKET_NAME_RAW_DATA")
    folder_name_raw_data = os.environ.get("FOLDER_NAME_RAW_DATA")
    return [feature_group_name, feature_version, partition_year, partition_month, partition_day,
            partition_hour, spark_config_setup, bucket, bucket_name_raw_data, folder_name_raw_data]


def pod_autoencoder_fe_input():
    """
    outputs
    -------
            list of parameters for pod rec type
            required for autoencoder feature engineering
            pipeline

    """
    ##feature parameters
    feature_group_name = "pod_autoencoder_ad"
    feature_version = "v0.0.2"

    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "9"
    partition_day = "9"
    partition_hour = "1"
    spark_config_setup = "384gb"

    ##s3 bucket parameters
    bucket = os.environ.get("BUCKET_NAME_OUTPUT")
    bucket_name_raw_data = os.environ.get("BUCKET_NAME_RAW_DATA")
    folder_name_raw_data = os.environ.get("FOLDER_NAME_RAW_DATA")
    return [feature_group_name, feature_version, partition_year, partition_month, partition_day,
            partition_hour, spark_config_setup, bucket, bucket_name_raw_data, folder_name_raw_data]


def pod_pca_fe_input():
    """
    outputs
    -------
            list of parameters for pod rec type
            required for pca feature engineering
            pipeline

    """
    ##feature parameters
    feature_group_name = "pod_pca_ad"
    feature_version = "v0.0.1"

    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "9"
    partition_day = "9"
    partition_hour = "1"
    spark_config_setup = "384gb"

    ##s3 bucket parameters
    bucket = os.environ.get("BUCKET_NAME_OUTPUT")
    bucket_name_raw_data = os.environ.get("BUCKET_NAME_RAW_DATA")
    folder_name_raw_data = os.environ.get("FOLDER_NAME_RAW_DATA")
    return [feature_group_name, feature_version,
            partition_year, partition_month, partition_day,
            partition_hour, spark_config_setup,
            bucket,bucket_name_raw_data,folder_name_raw_data]


def container_autoencoder_fe_input():
    """
    outputs
    -------
            list of parameters for container rec type
            required for autoencoder feature engineering
            pipeline

    """
    ##feature parameters
    feature_group_name = "container_autoencoder_ad"
    feature_version = "v0.0.1"

    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "5"
    partition_day = "5"
    partition_hour = "1"
    spark_config_setup = "384gb"

    ##s3 bucket parameters
    bucket = os.environ.get("BUCKET_NAME_OUTPUT")
    bucket_name_raw_data = os.environ.get("BUCKET_NAME_RAW_DATA")
    folder_name_raw_data = os.environ.get("FOLDER_NAME_RAW_DATA")
    return [feature_group_name, feature_version, partition_year, partition_month, partition_day,
            partition_hour, spark_config_setup, bucket, bucket_name_raw_data, folder_name_raw_data]


def container_pca_fe_input():
    """
    outputs
    -------
            list of parameters for container rec type
            required for pca feature engineering
            pipeline

    """
    ##feature parameters
    feature_group_name = "container_pca_ad"
    feature_version = "v0.0.1"

    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "5"
    partition_day = "5"
    partition_hour = "1"
    spark_config_setup = "384gb"

    ##s3 bucket parameters
    bucket = os.environ.get("BUCKET_NAME_OUTPUT")
    bucket_name_raw_data = os.environ.get("BUCKET_NAME_RAW_DATA")
    folder_name_raw_data = os.environ.get("FOLDER_NAME_RAW_DATA")
    return [feature_group_name, feature_version,
            partition_year, partition_month, partition_day,
            partition_hour, spark_config_setup,
            bucket,bucket_name_raw_data,folder_name_raw_data]
