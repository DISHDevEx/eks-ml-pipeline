"""Test the methods of the class S3Utilities in utilities/s3_utilities/py."""

from eks_ml_pipeline import S3Utilities
import numpy as np
import boto3

def test_upload_file(
    ae_train_input, # for instantiating the S3Utilities class
    bucket_name
    ):

    # generate a file to be uploaded
    np.save('test_upload_array.npy', np.array([1,2,3]))

    # Instantiate the class with fixtures from conftest.py.
    s3_util = S3Utilities(
        bucket_name = bucket_name,
        model_name = ae_train_input[1][0], #self.feature_selection[0] = feature_group_name
        version = ae_train_input[1][1], #self.feature_selection[1], # feature_input_version
        )

    # upload that file to s3 bucket specified in conftest

    # use s3_client.head_object(that file) to make sure the file is in s3
    # delete file from s3
    # check that file has been deleted.

    s3_util.upload_file(
        local_path = 'test_upload_array.npy',
        bucket_name = self.save_model_locations[1],
        key = "pytest_pca_ad/v0.0.1/data/data/tesnors/test_upload_array.npy",
        )


#                 self.s3_utilities.upload_file( # end of .join args

