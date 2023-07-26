import os
import pytest
from devex_sdk import get_features, EKS_Connector, Spark_Utils
from eks_ml_pipeline import AutoencoderModelDish5g
import tempfile
import shutil
import numpy as np
import pandas as pd
from eks_ml_pipeline import S3Utilities

def test_download_file(bucket_name):
    """Download with method under test, and check for file locally."""
    local_dir = tempfile.mkdtemp()
    local_fname = 'local_test_download_file.npy'
    local_path = f'{local_dir}/{local_fname}'

    # Instantiate the class with fixtures from conftest.py.
    s3_util = S3Utilities(
        bucket_name = bucket_name,
        model_name = '', # irrelevant for this test
        version = '' # irrelevant for this test
        )
    # download file
    s3_util.download_file(
        local_path = local_path,
        bucket_name = bucket_name,
        key = "pytest_s3_utilities/test_download_file.npy",
        )
    # check that file is local
    assert local_fname in os.listdir(local_dir)

    # Cleanup
    shutil.rmtree(local_dir) # remove tempdir