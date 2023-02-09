"""Test the methods of the class S3Utilities in utilities/s3_utilities.py."""
import os
from eks_ml_pipeline import S3Utilities
import numpy as np
import pandas as pd
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
    assert local_fname in os.listdir(local_dir)

    # delete local file
    os.remove(local_dir + local_fname)


def test_download_zip_and_unzip(
    ae_train_input, # for instantiating the S3Utilities class
    bucket_name
    ):

    # Instantiate the class with fixtures from conftest.py.
    s3_util = S3Utilities(
        bucket_name = bucket_name,
        model_name = 'pytest_s3_utilities',
        version = 'version',
        )

    # download file with method
    local_dir = './'
    local_fname = 'test_download_zip_file.zip'
    # from bucket/pytest_s3_utilities/version/folder/type/test_download_zip_file.zip
    s3_util.download_zip(
        local_path = local_dir + local_fname,
        folder = 'folder',
        type_ = "type",
        file_name = "test_download_zip_file.zip" # test data in s3
        )

    # check that file is local
    assert local_fname in os.listdir(local_dir)

    # test unzip method
    s3_util.unzip(path_to_zip = local_fname)

    # delete local_zip file
    os.remove(local_dir + local_fname)


# def test_unzip(): #1.0 does not run
#     ## generate two files to zip together
#     # create a direcory to zip
#     pre_zip_dir = 'pre_zip_test_files/'
#     os.makedirs(pre_zip_dir)
#     # populate directory with files
#     fnames = ['file1.txt','file2.txt']
#     for fname in fnames:
#         with open(pre_zip_dir + fname, 'a'):
#             os.utime(pre_zip_dir + fname)
#     # zip file to new directory
#     # post_zip_dir = 'zipped_test_files/'
#     # os.makedirs(post_zip_dir)
#     shutil.make_archive('zip_test_file',
#                         'zip',
#     #                     root_dir = post_zip_dir,
#     #                     base_dir = post_zip_dir
#                        )
#     # clean up
#     for fname in fnames:
#         os.remove(pre_zip_dir + fname)
#     os.rmdir(pre_zip_dir)

#     ## unip with the method under test
#     # Instantiate the class with fixtures from conftest.py.
#     s3_util = S3Utilities(
#         bucket_name = bucket_name,
#         model_name = ae_train_input[1][0], #self.feature_selection[0] = feature_group_name
#         version = ae_train_input[1][1], #self.feature_selection[1], # feature_input_version
#         )
#     s3_util.unzip(path_to_zip = 'zip_test_file.zip' )

#     # delete local zip file
#     os.remove('zip_test_file.zip')
#     # delete unzipped directory
#     os.rmdir('zip_test_file')


# def test_zip_and_upload():
    # generate local file
    # zip and upload to s3 with method
    # check that file is in s3
    # delete uploaded file from s3

def test_pandas_dataframe_to_s3(bucket_name):
#     # create pandas df in memory
    data = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=data)
    # upload to s3 with method
    s3_util = S3Utilities(
        bucket_name = bucket_name,
        model_name = 'pytest_s3_utilities',
        version = 'version',
        )
#     pandas_file_name = 'test_pandas_to_s3.parquet'
    pandas_file_name = 'test_read_parquet_to_pandas_df.parquet'
    s3_util.pandas_dataframe_to_s3(
        input_datafame = df, 
        folder = 'folder', 
        type_  = "type", 
        file_name = pandas_file_name
        )

#     # check that file is in s3
#     s3_util.client.head_object(
#         Bucket=bucket_name,
#         Key = "pytest_s3_utilities/version/folder/type/" + pandas_file_name
#         )


#     # delete file from s3
#     s3_util.client.delete_object(
#         Bucket=bucket_name,
#         Key = "pytest_s3_utilities/version/folder/type/" + pandas_file_name
#         )


def test_write_tensor(bucket_name):
    # generate tensor
    test_tensor = np.zeros((2,3,4,5))
    # write to s3 with method
    s3_util = S3Utilities(
        bucket_name = bucket_name,
        model_name = 'pytest_s3_utilities',
        version = 'version',
        )
#     numpy_file_name = 'test_numpy_to_s3.npy'
    numpy_file_name = 'test_read_tensor.npy'
    s3_util.write_tensor(
        tensor = test_tensor, 
        folder = 'folder', 
        type_  = "type", 
        file_name = numpy_file_name
        )
#     # check that file is in s3
#     s3_util.client.head_object(
#         Bucket=bucket_name,
#         Key = "pytest_s3_utilities/version/folder/type/" + numpy_file_name
#         )

#     # delete file from s3
#     s3_util.client.delete_object(
#         Bucket=bucket_name,
#         Key = "pytest_s3_utilities/version/folder/type/" + numpy_file_name
#         )

def test_awswrangler_pandas_dataframe_to_s3(bucket_name):
    # ??how is this different than pandas_dataframe_to_s3?

    # generate pandas df in memory
    data = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=data)
    # upload to s3 with the method under test
    # Instantiate object
    s3_util = S3Utilities(
        bucket_name = bucket_name,
        model_name = 'pytest_s3_utilities',
        version = 'version',
        )
    pandas_file_name = 'test_wrangler_pandas_to_s3.parquet'
    s3_util.awswrangler_pandas_dataframe_to_s3(
        tensor = test_tensor, 
        folder = 'folder', 
        type_  = "type", 
        file_name = numpy_file_name
        )
    # check that file is in s3
    s3_util.client.head_object(
        Bucket=bucket_name,
        Key = "pytest_s3_utilities/version/folder/type/" + numpy_file_name
        )
    # delete file from s3
    s3_util.client.delete_object(
        Bucket=bucket_name,
        Key = "pytest_s3_utilities/version/folder/type/" + numpy_file_name
        )
    
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

