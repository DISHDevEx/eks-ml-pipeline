import numpy as np
import pandas as pd
import awswrangler as wr
import ast
from pandas import json_normalize
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from devex_sdk import Pyspark_data_ingestion, get_features
from .utilities import S3Utilities
from .inputs import training_input, inference_input
from .train_test_pipelines import TrainTestPipelines


"""
Contributed by Evgeniya Dontsova and Praveen Mada
MSS Dish 5g - Pattern Detection

this inference functions will be used for Anomaly Detection models
"""


def only_dict(d):
    """
    inputs
    ------
            d: dict

    outputs
    -------
            Convert json string representation of dictionary to a python dict

    """
    return ast.literal_eval(d)

def inference_data_naming(input_year, input_month, input_day, input_hour, rec_type):

    """
    inputs
    ------
            input_year : STRING | Int
            the year from which to read data, leave empty for all years

            input_month : STRING | Int
            the month from which to read data, leave empty for all months

            input_day : STRING | Int
            the day from which to read data, leave empty for all days

            input_hour: STRING | Int
            the hour from which to read data, leave empty for all hours

            rec_type: STRING
            uses schema for rec type when building pyspark df

    outputs
    -------
            file_name: str
            file name for raw inference data

    """


    if input_hour == -1:
        file_name = f'{rec_type}/{rec_type}_{input_year}_{input_month}_{input_day}'
    elif input_day == -1:
        file_name = f'{rec_type}/{rec_type}_{input_year}_{input_month}'
    else:
        file_name = f'{rec_type}/{rec_type}_{input_year}_{input_month}_{input_day}_{input_hour}'

    return file_name


def inference_data_builder(input_year, input_month, input_day, input_hour, rec_type,
                           input_setup, bucket):

    """
    inputs
    ------
            input_year : STRING | Int
            the year from which to read data, leave empty for all years

            input_month : STRING | Int
            the month from which to read data, leave empty for all months

            input_day : STRING | Int
            the day from which to read data, leave empty for all days

            input_hour: STRING | Int
            the hour from which to read data, leave empty for all hours

            rec_type: STRING
            uses schema for rec type when building pyspark df

            input_setup: STRING
            kernel config

    outputs
    -------
            writes parquet to specific s3 path
            outputs s3 path for written parquet file
    """

    file_name = inference_data_naming(input_year, input_month,
                                      input_day, input_hour, rec_type)

    pyspark_data = Pyspark_data_ingestion(year = input_year, month = input_month,
                                          day = input_day, hour = input_hour,
                                          setup = input_setup, filter_column_value = rec_type)

    err, pyspark_df = pyspark_data.read()
    print(err)

    raw_data_s3_path = False

    if err == 'PASS':
        raw_data_s3_path = f"s3://{bucket}/inference_data/{file_name}.parquet"
        pyspark_df = pyspark_df.toPandas()
        wr.s3.to_parquet(pyspark_df, path=raw_data_s3_path)

    return raw_data_s3_path


def build_processed_data(inference_input_parameters,
                         training_input_parameters):

    """
    inputs
    ------
            inference_input_parameters: list
            list of parameters required for inference

            training_input_parameters: list
            list of training input parameter specific to rec type and model


    outputs
    -------
            training_input_parameters: list
            updated list of training input parameter specific to rec type and model

    """

    ### Load input parameters

    #inference input
    rec_type, sampling_column, \
    partition_year, partition_month, partition_day, partition_hour, \
    spark_config_setup, data_bucketname = inference_input_parameters

    #training input
    encode_decode_model = training_input_parameters[0]
    feature_group_name, feature_input_version = training_input_parameters[1]
    data_bucketname, train_data_filename, test_data_filename = training_input_parameters[2]
    save_model_local_path, model_bucketname, model_filename = training_input_parameters[3]


    #raw inference data path
    raw_inference_file_name = inference_data_naming(input_year = partition_year,
                                                    input_month = partition_month,
                                                    input_day = partition_day,
                                                    input_hour = partition_hour,
                                                    rec_type = rec_type)

    raw_data_s3_path = f"s3://{data_bucketname}/inference_data/{raw_inference_file_name}.parquet"

    ###Initialize s3 utilities class
    s3_utils = S3Utilities(bucket_name = data_bucketname,
                           model_name = feature_group_name,
                           version = feature_input_version)


    ###Load data: Read raw data in parquet format from s3
    print(f"\n*** Reading raw data from: {raw_data_s3_path}***\n")

    try:
        df = pd.read_parquet(raw_data_s3_path)
    except:
        print(f"\n*** Raw data being build in {raw_data_s3_path}***\n")

        #Build raw inference data and get the s3 path
        raw_data_s3_path = inference_data_builder(input_year = partition_year, input_month = partition_month,
                                                  input_day = partition_day, input_hour = partition_hour,
                                                  rec_type = rec_type, input_setup = spark_config_setup,
                                                  bucket = data_bucketname)
        df = pd.read_parquet(raw_data_s3_path)

    ### Process raw data
    #Read features and parameters
    features_df = get_features(feature_group_name, feature_input_version)
    features = features_df["feature_name"].to_list()
    #remove spaces: that were put by mistake
    features = [feature.strip(' ') for feature in features]
    model_parameters = features_df["model_parameters"].iloc[0]
    time_steps = model_parameters["time_steps"]

    #drop null values
    df = df.dropna(subset = features, axis = 0)

    #check sampling column: if needs unpacking
    if type(sampling_column) == tuple:
        #unpack
        df_unpack = json_normalize(df[sampling_column[0]].apply(only_dict)\
                                  .tolist()).add_prefix(sampling_column[0] + '.')
        df = df.join(df_unpack)
        sampling_column = '.'.join(sampling_column)

    sampling_column_unique = df[sampling_column].unique()

    if len(sampling_column_unique) >=1:
        #select unique sampling_column (e.g. InstanceId for Node or pod_id for Pod
        random_id = np.random.choice(df[sampling_column].unique())
        print(f'\n*** Select data with unique {sampling_column} = {random_id} ***\n')
        df = df.loc[(df[sampling_column] == random_id)]
        #sort by time
        df = df.sort_values(by='Timestamp').reset_index(drop=True)

        #select last time slice of data
        start = df.shape[0] - time_steps
        df = df.loc[start:start+time_steps, features]

        print("\n***** Inference input data shape*****")
        print(df.shape)
#         print("\n*** Inference data tensor ***")
#         print(df)
        print("\n***************************************\n")

        #scaler transformations
        scaler = StandardScaler()
        scaled_features = ["scaled_" + feature for feature in features]
        df[scaled_features] = scaler.fit_transform(df[features])
        inference_input_tensor = np.expand_dims(df[scaled_features], axis = 0)

        print("\n***** Inference input tensor shape*****")
        print(inference_input_tensor.shape)
#         print("\n*** Inference input tensor ***")
#         print(inference_input_tensor)
        print("\n***************************************\n")

        sampling_column = sampling_column.split('.')[-1]
        saved_file_name = ('_').join(['inference', sampling_column, random_id]) + ".npy"

        s3_utils.write_tensor(tensor = inference_input_tensor,
                              folder = "data",
                              type_ = "tensors",
                              file_name = saved_file_name)

        #update training input parameters with new test_data_filename
        training_input_parameters[2][1] = saved_file_name

    else:
        print(f"Exception occured: no unique values for {sampling_column} column.")

    return training_input_parameters


def inference_pipeline(inference_input_parameters,
                       training_input_parameters):

    """
    Parameters
    ----------
    inference_input_parameters: list
        list of parameters required for inference

    training_input_parameters: list
        list of training input parameter specific to rec type and model

    prediction_pipeline: function
        available values are autoencoder_testing_pipeline() or pca_testing_pipeline()

    Returns
    -------
    predictions: np.array
        model predictions

    residuals: np.array
        model residuals

    """

    # replace test data with data for inference.
    training_input_parameters = build_processed_data(
        inference_input_parameters, training_input_parameters)

    TrainTestPipelines(training_input_parameters).test()

    return None
