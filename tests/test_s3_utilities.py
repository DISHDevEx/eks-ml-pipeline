"""Test the methods of the class S3Utilities in utilities/s3_utilities.py."""
import os
from eks_ml_pipeline import S3Utilities
import numpy as np
import pandas as pd
import boto3
import tempfile
import shutil

def test_upload_file(
    ae_train_input, # for instantiating the S3Utilities class
    bucket_name
    ):
    """Generate file, upload, and check for uploaded file."""

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

    # delete file from s3
    s3_util.client.delete_object(
        Bucket=bucket_name,
        Key = "pytest_s3_utilities/" + filename
        )

def test_download_file(
    ae_train_input, # for instantiating the S3Utilities class
    bucket_name
    ):
    """Download with method under test, and check for file locally."""
    local_dir = tempfile.mkdtemp()
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

    # download file with method under test
    tmpdir = tempfile.mkdtemp()
    local_fname = 'local_test_download_zip_file.zip' # what to locally name .zip file
    local_path = f'{tmpdir}/{local_fname}'  # local destination + name
    s3_fname = "test_download_zip_file.zip" # test data in s3 
    s3_util.download_zip(
            local_path = local_path,
            folder = 'folder', # directory in s3 bucket
            type_ = "type", # directory in s3 bucket
            file_name = s3_fname # test data in s3 
            )
    # test .download_zip method
    # i.e. that the .zip file is local
    assert local_fname in os.listdir(tmpdir)

    # test .unzip method
    # i.e. that sfile is been extracted
    s3_util.unzip(path_to_zip = local_path)
    unziped_dir = tmpdir # directory where the files are placed
    file_list = os.listdir(unziped_dir) # files in that directory 
    # Two files were generated for this test and zipped into s3_fname
    # 'test_download_zip_file_1.txt' and 'test_download_zip_file_2.txt' 
    test_file_names = ['test_download_zip_file_1.txt',
                       'test_download_zip_file_2.txt']
    for file_name in test_file_names:
        assert file_name in file_list

    # clean up
    shutil.rmtree(tmpdir) # remove tempdir

def test_zip_and_upload(bucket_name):
    # generate local file in a tempdir
    tmpdir = tempfile.mkdtemp()
    fp = f'{tmpdir}/test_file_to_zip.txt'
    open(fp, 'w')
    # zip and upload to s3 with method
    s3_util = S3Utilities(
        bucket_name = bucket_name,
        model_name = 'pytest_s3_utilities',
        version = 'version',
        )
    s3_filename = "test_upload_zip_file.zip"
    s3_util.zip_and_upload(
        local_path = tmpdir, #directory containing the files to zip
        folder = 'folder',
        type_ = "type",
        file_name = s3_filename 
        )
    # check that file is in s3
        # check that file is in s3
    s3_util.client.head_object(
        Bucket=bucket_name,
        Key = "pytest_s3_utilities/version/folder/type/" + s3_filename
        )

    # delete uploaded file from s3
    s3_util.client.delete_object(
        Bucket=bucket_name,
        Key = "pytest_s3_utilities/version/folder/type/" + s3_filename
        )
    
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
    pandas_file_name = 'test_pandas_to_s3.parquet' #custom test data
    s3_util.pandas_dataframe_to_s3(
        input_datafame = df, 
        folder = 'folder', 
        type_  = "type", 
        file_name = pandas_file_name
        )
    # check that file is in s3
    s3_util.client.head_object(
        Bucket=bucket_name,
        Key = "pytest_s3_utilities/version/folder/type/" + pandas_file_name
        )

    # delete file from s3
    s3_util.client.delete_object(
        Bucket=bucket_name,
        Key = "pytest_s3_utilities/version/folder/type/" + pandas_file_name
        )


def test_write_tensor(bucket_name):
    # generate tensor
    test_tensor = np.zeros((2,3,4,5))
    # write to s3 with method
    s3_util = S3Utilities(
        bucket_name = bucket_name,
        model_name = 'pytest_s3_utilities',
        version = 'version',
        )
    numpy_file_name = 'test_numpy_to_s3.npy'
    s3_util.write_tensor(
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

def test_awswrangler_pandas_dataframe_to_s3(bucket_name):
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
        input_datafame = df, 
        folder = 'folder', 
        type_  = "type", 
        file_name = pandas_file_name
        )
    # check that file is in s3
    s3_util.client.head_object(
        Bucket=bucket_name,
        Key = "pytest_s3_utilities/version/folder/type/" + pandas_file_name
        )
    # delete file from s3
    s3_util.client.delete_object(
        Bucket=bucket_name,
        Key = "pytest_s3_utilities/version/folder/type/" + pandas_file_name
        )
    
def test_read_tensor(bucket_name):
    # instantiate object to be tested
    s3_util = S3Utilities(
        bucket_name = bucket_name,
        model_name = 'pytest_s3_utilities',
        version = 'version',
        )
    # read test tensor with method under test
    numpy_file_name = 'test_read_tensor.npy'
    test_tensor = s3_util.read_tensor(
        folder = 'folder', 
        type_  = "type", 
        file_name = numpy_file_name)
    # check that tensor is in memory
    test_tensor.shape 

def test_upload_directory(bucket_name):
    # create directory with two files in it
    test_directory = 'test_directory/'
    os.makedirs(test_directory)
    # populate directory with files
    fnames = ['file1.txt','file2.txt']
    for fname in fnames:
        with open(test_directory + fname, 'a'):
            os.utime(test_directory + fname)
    # instantiate object under test 
    s3_util = S3Utilities(
        bucket_name = bucket_name,
        model_name = 'pytest_s3_utilities',
        version = 'version',
        )
    # use method under test to upload dir with contents
    s3_util.upload_directory(
        local_path = test_directory, 
        folder = 'folder', 
        type_  = "type", 
    )
    # use s3_client.head_object(that file) to make sure the file is in s3
    for fname in fnames:
        s3_util.client.head_object(
            Bucket = bucket_name,
            Key = ("pytest_s3_utilities/version/folder/type/" 
                  + fname)
            )

    # delete files from s3
    for fname in fnames:
        s3_util.client.delete_object(
            Bucket=bucket_name,
            Key = ("pytest_s3_utilities/version/folder/type/" 
                  + fname)
            )

#### NOTE ####      
# test_pyspark_write_parquet() need not exist 
# because it would test only a pyspark funcion.  

    
def test_read_parquet_to_pandas_df(bucket_name):
    # instantiate object to be tested
    s3_util = S3Utilities(
        bucket_name = bucket_name,
        model_name = 'pytest_s3_utilities',
        version = 'version',
        )
    # read test parquet with method under test
    parquet_file_name = 'test_read_parquet_to_pandas_df.parquet'
    test_df = s3_util.read_parquet_to_pandas_df(
        folder = 'folder', 
        type_  = "type", 
        file_name = parquet_file_name)
    # check that tensor is in memory
    test_df.shape 
