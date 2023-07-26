import os
import pytest
from devex_sdk import get_features, EKS_Connector, Spark_Utils
from eks_ml_pipeline import AutoencoderModelDish5g
import tempfile
import shutil
import numpy as np
import pandas as pd
from eks_ml_pipeline import S3Utilities


    
def test_write_tensor(bucket_name):
    """Create tensor, upload with method under test, and check for file in S3"""
    # Instantiate the class with fixtures from conftest.py.
    s3_util = S3Utilities(
        bucket_name = bucket_name,
        model_name = 'pytest_s3_utilities', # destination dir
        version = 'version', # destination dir
        )

    # generate tensor
    test_tensor = np.zeros((2,3,4,5))

    # write to s3 with method
    numpy_file_name = 'test_numpy_to_s3.npy'
    s3_util.write_tensor(
        tensor = test_tensor,
        folder = 'folder', # destination dir
        type_  = "type", # destination dir
        file_name = numpy_file_name
        )

    # check that file is in s3
    head = s3_util.client.head_object(
        Bucket=bucket_name,
        Key = "pytest_s3_utilities/version/folder/type/" + numpy_file_name
        )
    # HTTP status code 200 indicates request succeeded
    assert head['ResponseMetadata']['HTTPStatusCode'] == 200

    # delete file from s3
    s3_util.client.delete_object(
        Bucket=bucket_name,
        Key = "pytest_s3_utilities/version/folder/type/" + numpy_file_name
        )