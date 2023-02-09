"""Test the methods of the class S3Utilities in utilities/s3_utilities.py."""
import os
from eks_ml_pipeline import S3Utilities
import numpy as np
import boto3

def test_upload_file(
    ae_train_input, # for instantiating the S3Utilities class
    bucket_name
    ):

    # generate a file to be uploaded
    filename = 'test_upload_file.npy'
    np.save(filename, np.array([1,2,3]))

    # Instantiate the class with fixtures from conftest.py.
    s3_util = S3Utilities(
        bucket_name = bucket_name,
        model_name = ae_train_input[1][0], #self.feature_selection[0] = feature_group_name
        version = ae_train_input[1][1], #self.feature_selection[1], # feature_input_version
        )

    # upload that file to s3 bucket specified in conftest
    s3_util.upload_file(
        local_path = filename,
        bucket_name = bucket_name,
        key = "pytest_s3_utilities/" + filename
        )

    # use s3_client.head_object(that file) to make sure the file is in s3
    s3_util.client.head_object(
        Bucket=bucket_name,
        Key = "pytest_s3_utilities/" + filename
        )

    # delete file from s3
    s3_util.client.delete_object(
        Bucket=bucket_name,
        Key = "pytest_s3_utilities/" + filename
        )


#     # check that file has been deleted.
#     try:
#         s3_util.client.head_object(
#             Bucket=bucket_name,
#             Key = "pytest_s3_utilities/" + filename
#             )
#     except ClientError:
#         print('Test file sucessfully deleted')


def test_download_file(
    ae_train_input, # for instantiating the S3Utilities class
    bucket_name
    ):

    local_dir = './'
    local_fname = 'test_download_file.npy'

    # Instantiate the class with fixtures from conftest.py.
    s3_util = S3Utilities(
        bucket_name = bucket_name,
        model_name = ae_train_input[1][0], #self.feature_selection[0] = feature_group_name
        version = ae_train_input[1][1], #self.feature_selection[1], # feature_input_version
        )
    # download file
    s3_util.download_file(
        local_path = local_dir + local_fname,
        bucket_name = bucket_name,
        key = "pytest_s3_utilities/test_download_file.npy",
        )
    # check that file is local
    assert local_fname in os.listdir(local_path)

    # delete local file
    os.remove(local_dir + local_fname)


# def test_download_zip(
#     ae_train_input, # for instantiating the S3Utilities class
#     bucket_name
#     ):

#     # Instantiate the class with fixtures from conftest.py.
#     s3_util = S3Utilities(
#         bucket_name = bucket_name,
#         model_name = ae_train_input[1][0], #self.feature_selection[0] = feature_group_name
#         version = ae_train_input[1][1], #self.feature_selection[1], # feature_input_version
#         )

#     # download file

#     # check that file is local

#     # delete local file

# def test_unzip():
#     # generate local zip file
#     # unip with the method
#     # delete local zip file


# def test_zip_and_upload():
#     # generate local file
#     # zip and upload to s3 with method
#     # check that file is in s3
#     # delete uploaded file from s3

# def test_pandas_dataframe_to_s3():
#     # create pandas df in memory
#     # upload to s3 with method
#     # check that file is in s3
#     # delete uploaded file from s3

# def test_write_tensor():
#     # generate tensor
#     # write to s3 with method
#     # check that file is in s3
#     # delete uploaded file from s3

# def test_awswrangler_pandas_dataframe_to_s3():
#     # ??how is this different than pandas_dataframe_to_s3?

#     # generate pandas df in memory
#     # upload to s3
#     # check that file is in s3
#     # delete uploaded file from s3

# def test_read_tensor():
#     # read test tensor
#     # check that tensor is in memory

# def test_upload_directory():
#     # create directory with two files in it
#     # use method to upload dir with contents
#     # check that both files are present
#     # delete dir from s3

# def test_pyspark_write_parquet():
#     # create pyspark dataframe
#     # use method to upload to s3
#     # check that file is in s3
#     # delete file from s3

# def test_read_parquet_to_pandas_df():
#     # read test parquet file with method
#     # check that there is a pandas data frame

