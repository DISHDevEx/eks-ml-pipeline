import os
import pytest
from devex_sdk import get_features, EKS_Connector, Spark_Utils
from eks_ml_pipeline import AutoencoderModelDish5g
import tempfile
import shutil
import numpy as np
import pandas as pd
from eks_ml_pipeline import S3Utilities


@pytest.fixture(scope="module")
def bucket_name():
    """
    Get bucket name from the github workflow runner secrets
    """
    BUCKET_NAME = os.environ.get("BUCKET_NAME_PYTEST") 
    return BUCKET_NAME


def test_ae_train_input(bucket_name):
    """
    Create inputs to train the desired model.
    Includes all of bucket versioning and model versioning needed
    as well as the file locations for a pipeline.
    Parameters
    ----------
    None

    Returns
    -------
    training_inputs
        List of parameters for node rec type
        required by autoencoder model
        training pipeline
    """

    # *****************************************************#
    # ********** data and model input parameters **********#
    # *****************************************************#

    # feature_selection
    feature_group_name = "pytest_autoencoder_ad"
    feature_input_version = "v0.0.1"

    # data_locations
    data_bucketname = bucket_name
    train_data_filename = "aeDummyDataTrain.npy"
    test_data_filename = "aeDummyDataTest.npy"

    # save_model_locations
    save_model_local_path = "../test_autoencoder"
    model_bucketname = bucket_name
    model_name = "train_autoencoder_ad"
    model_version = "v0.0.0Pytest"

    # Define model filename and path
    model_filename = "_".join(
        [
            model_name,
            "model",
            model_version,
            train_data_filename.split(".")[-2],  # all preceeding extension
        ]
    )
    
    print('Model Filename:', model_filename)

    # ********************************************#
    # ********** initialize model class **********#
    # ********************************************#

    features_df = get_features(feature_group_name, feature_input_version)
    
    print(features_df.columns)
    print(features_df.count())
    
    print (features_df["model_parameters"].iloc[0])
    
    assert len(features_df.columns) == 11


# def test_upload_file(
#     ae_train_input, # for instantiating the S3Utilities class
#     bucket_name
#     ):
#     """Upload file wtih method under test, and check for uploaded file."""

#     # generate a file to be uploaded and save it in a temp path
#     filename = 'test_upload_file.npy'
#     tmpdir = tempfile.mkdtemp()
#     file_path = f'{tmpdir}/{filename}'

#     np.save(file_path, np.array([1,2,3]))

#     # Instantiate the class with fixtures from conftest.py.
#     s3_util = S3Utilities(
#         bucket_name = bucket_name,
#         model_name = ae_train_input[1][0], #feature_selection[0] = feature_group_name
#         version = ae_train_input[1][1], #feature_selection[1] =  feature_input_version
#         )

#     # use the method under test to upload that file to s3
#     s3_util.upload_file(
#         local_path = file_path,
#         bucket_name = bucket_name,
#         key = "pytest_s3_utilities/" + filename
#         )

#     # test that the file is in s3
#     head = s3_util.client.head_object(
#         Bucket = bucket_name,
#         Key = "pytest_s3_utilities/" + filename
#         )
#     # HTTP status code 200 indicates request succeeded
#     assert head['ResponseMetadata']['HTTPStatusCode'] == 200

#     # Cleanup by deleting file from s3.
#     s3_util.client.delete_object(
#         Bucket=bucket_name,
#         Key = "pytest_s3_utilities/" + filename
#         )