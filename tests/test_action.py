import os
import pytest
from devex_sdk import get_features, EKS_Connector, Spark_Utils
from eks_ml_pipeline import AutoencoderModelDish5g
import tempfile
import shutil
import numpy as np
import pandas as pd
from eks_ml_pipeline import S3Utilities


def test_upload_file(
    ae_train_input, # for instantiating the S3Utilities class
    bucket_name
    ):
    """Upload file wtih method under test, and check for uploaded file."""

    # generate a file to be uploaded and save it in a temp path
    filename = 'test_upload_file.npy'
    tmpdir = tempfile.mkdtemp()
    file_path = f'{tmpdir}/{filename}'

    np.save(file_path, np.array([1,2,3]))

    # Instantiate the class with fixtures from conftest.py.
    s3_util = S3Utilities(
        bucket_name = bucket_name,
        model_name = ae_train_input[1][0], #feature_selection[0] = feature_group_name
        version = ae_train_input[1][1], #feature_selection[1] =  feature_input_version
        )

    # use the method under test to upload that file to s3
    s3_util.upload_file(
        local_path = file_path,
        bucket_name = bucket_name,
        key = "pytest_s3_utilities/" + filename
        )

    # test that the file is in s3
    head = s3_util.client.head_object(
        Bucket = bucket_name,
        Key = "pytest_s3_utilities/" + filename
        )
    # HTTP status code 200 indicates request succeeded
    assert head['ResponseMetadata']['HTTPStatusCode'] == 200

    # Cleanup by deleting file from s3.
    s3_util.client.delete_object(
        Bucket=bucket_name,
        Key = "pytest_s3_utilities/" + filename
        )