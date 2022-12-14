from io import BytesIO
import numpy as np
from urllib.parse import urlparse
import boto3
import os
import shutil
import zipfile


def read_tensor(bucket_name, model_name,version, model_data_type):
    """
    inputs
    ------
            bucket_name: STRING
            s3 bucket name to read tensor from

             bucket_name: STRING
            s3 bucket name to write the tensor

            model_name: STRING
            model name to create a folder within the bucket for model versioning

            version: STRING
            format: v#.#.#
            version will be used versioning

    outputs
    -------
            tensor : numpy tensor

    """
    print(f"reading tensor from: {bucket_name}/{model_name}/{version}/data/tensors/{model_data_type}.npy")
    client = boto3.client('s3')
    bytes_ = BytesIO()
    client.download_fileobj(Fileobj=bytes_, Bucket=bucket_name, Key=f'{model_name}/{version}/data/tensors/{model_data_type}.npy')
    bytes_.seek(0)
    tensor = np.load(bytes_, allow_pickle=True)

    return tensor




def write_tensor(tensor, bucket_name, model_name, version, flag, file_name):
    """
    inputs
    ------
            tensor: numpy array
            numpy array stored in a python variable

            bucket_name: STRING
            s3 bucket name to write the tensor

            model_name: STRING
            model name to create a folder within the bucket for model versioning

            version: STRING
            format: v#.#.#
            version will be used versioning

            model_data_type: string
            This is the string to test weather its training or testing data
    outputs
    -------
            path : string
            path where will the tensor is stored in s3

    """
    client = boto3.client('s3')
    bytes_ = BytesIO()
    np.save(bytes_, tensor, allow_pickle=True)
    bytes_.seek(0)
    # client.put_object(Body=a, Bucket=bucket, Key='array.npy')
    
    if flag == "data":
        client.upload_fileobj(Fileobj=bytes_, Bucket=bucket_name,
                             Key=f'{model_name}/{version}/data/tensors/{file_name}.npy')
        path = f'{bucket_name}/{model_name}/{version}/data/tensors/{file_name}.npy'
    if flag == "model":
        client.upload_fileobj(Fileobj=bytes_, Bucket=bucket_name,
                             Key=f'{model_name}/{version}/model/{file_name}.npy')
        path = f'{bucket_name}/{model_name}/{version}/model/{file_name}.npy'
    return path


def uploadDirectory(local_path, bucketname, model_name, version):
    """
    inputs
    ------
            local_path: string
            local path to folder NOT file to upload the folder to s3

            bucket_name: STRING
            s3 bucket name to write the folder

            model_name: STRING
            model name to create a folder within the bucket for model versioning

            version: STRING
            format: v#.#.#
            version will be used versioning


    outputs
    -------
            prints that the upload has completed

    """
    for root, dirs, files in os.walk(local_path):
        for file in files:
            client = boto3.client('s3')
            client.upload_file(os.path.join(root, file), bucketname, model_name + '/' + version + '/models/' + file)
    print(f"file uploaded too: {bucketname}/{model_name}/{version}/{models}/{file}")

def write_parquet(df, bucket_name, model_name, version, model_data_type):
    """
    inputs
    ------
            df: pysaprk dataframe
            pyspark data fram to be written to s3 in partations

            bucket_name: STRING
            s3 bucket name to write the parquet to


            model_name: STRING
            model name to create a folder within the bucket for model versioning


            version: STRING
            format: v#.#.#
            version will be used versioning


            model_data_type: string
            This is the string to test weather its training or testing data



    outputs
    -------
            prints that the upload has completed

    """

    df.write.mode('overwrite').parquet(f's3a://{bucket_name}/{model_name}/{version}/data/pyspark/{model_data_type}/')

def upload_zip(local_path, bucket_name, model_name, version, file):
    """
    inputs
    ------
            local_path: string
            local path to folder NOT file to upload the folder to s3

            bucket_name: STRING
            s3 bucket name to write the folder

            model_name: STRING
            model name to create a folder within the bucket for model versioning

            version: STRING
            format: v#.#.#
            version will be used versioning


    outputs
    -------
            prints that the upload has completed

    """
    path = shutil.make_archive(local_path, 'zip', local_path)
    client = boto3.client('s3')
    client.upload_file(path, bucket_name, model_name + '/' + version + '/models/' + file + ".zip")
    print(f"Zip file uploaded: {bucket_name}/{model_name}/{version}/models/{file}.zip")


def download_zip(download_path, bucket_name, model_name, version, file):
    """
    inputs
    ------
            download_path: string
            local path where to download the file too.

            bucket_name: STRING
            s3 bucket name to download the file from

            model_name: STRING
            model name to create a folder within the bucket for model versioning

            version: STRING
            format: v#.#.#
            version will be used for versioning


    outputs
    -------
            prints that the upload has completed

    """

    with open(download_path, 'wb') as f:
        client = boto3.client('s3')
        client.download_fileobj(bucket_name, model_name + '/' + version + '/models/' + file +".zip", f)
    print("zip file downloaded to : ")

    
def unzip(path_to_zip, extract_location):
    """
    inputs
    ------
            path_to_zip: string
            local path where the zip file is located

            extract_location: STRING
            location where the files need to be extracted


    outputs
    -------
            prints that the upload has completed

    """
    with zipfile.ZipFile(path_to_zip, "r") as zip_ref:
        zip_ref.extractall(extract_location)
    print("file unzipped")
    
    
    
    
    
def pandas_dataframe_to_s3(input_datafame, bucket_name, model_name, version, model_data_type):
    """
    inputs
    ------
            input_datafame: pandas dataframe
            pandas dataframe variable

            bucket_name: STRING
            s3 bucket name to write the parquet to


            model_name: STRING
            model name to create a folder within the bucket for model versioning


            version: STRING
            format: v#.#.#
            version will be used versioning


            model_data_type: string
            This is the string to differentiate weather its training or testing data


    outputs
    -------
            prints that the upload has completed

    """
    client = boto3.client('s3')
    out_buffer = BytesIO()
    input_datafame.to_parquet(out_buffer, index=False)
    client.put_object(Bucket=bucket_name, Key=f"{model_name}/{version}/data/pandas_df/{model_data_type}.parquet", Body=out_buffer.getvalue())
    return print("sucess")



def awswrangler_pandas_dataframe_to_s3(input_datafame, bucket_name, model_name, version, model_data_type):
    """
    inputs
    ------
            input_datafame: pandas dataframe
            pandas dataframe variable
            
             bucket_name: STRING
            s3 bucket name to write the parquet to


            model_name: STRING
            model name to create a folder within the bucket for model versioning


            version: STRING
            format: v#.#.#
            version will be used versioning


            model_data_type: string
            This is the string to differentiate weather its training or testing data

    outputs
    -------
            prints that the upload has completed

    """
    import awswrangler as wr
    wr.s3.to_parquet(input_datafame,path=f"s3://{bucket_name}/{model_name}/{version}/data/pandas_df/{model_data_type}.parquet")
    return print("sucess")

