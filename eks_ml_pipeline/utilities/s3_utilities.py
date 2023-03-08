"""Utilities file to upload and downlod files from s3."""
import os
import zipfile
from io import BytesIO
import shutil
import logging
import numpy as np
import pandas as pd
import boto3
import awswrangler as wr
from botocore.exceptions import ClientError


class S3Utilities:
    """
    inputs
    ------
            bucket_name: STRING
            s3 bucket name to read from

            model_name: STRING
            model name to create a folder within the bucket for model versioning

            version: STRING
            format: v#.#.#
            version will be used versioning

            ####################
            All the above parameters are class variables
            ###################

            local_path: STRING
            local path for reading or writing files


            key: STRING
            entire path to s3 for readind and writing files.


            folder: STRING
            folder name to differentiate between models or data etc

            type_: STRING
            type name to differentiate between tensors or pandas_parquet etc

            file_name: STRING
            name of the file with extention


    outputs
    -------
            upload_file(local_path, bucket_name, key): path
            download_file(local_path, bucket_name, key): path
            download_zip(writing_path, folder, type_, file_name): path
            unzip(path_to_zip, extract_location): path
            zip_and_upload(local_path, folder, type_, file_name): path
            pandas_dataframe_to_s3(input_datafame, folder, type_, file_name): path
            write_tensor(tensor, folder, type_, file_name): path
            awswrangler_pandas_dataframe_to_s3(input_datafame,folder, type_, file_name): path
            read_tensor(folder, type_, file_name): variable: numpy tensor
            upload_directory(local_path, folder, type_): path
            pyspark_write_parquet(df,folder, type_): path
            read_parquet_to_pandas_df(folder, type_, file_name): dataframe
    """

    def __init__(self, bucket_name=None, model_name=None, version=None):
        """
        class constructor.
        """
        self.bucket_name = bucket_name
        self.model_name = model_name
        self.version = version
        self.client = boto3.client('s3')
        self.resource = boto3.resource('s3')
        if self.bucket_name is None or self.model_name is None or self.version is None:
            raise Exception("Please Initialize Class parameters")

    def upload_file(self, local_path, bucket_name, key):
        """
        upload any file to s3.
        """
        try:
            self.client.upload_file(local_path, bucket_name, key)
        except ClientError as error:
            logging.error(error)
            return False
        return print(f'Uploaded Path: s3://{bucket_name}/{key}')

    def download_file(self, local_path, bucket_name, key):
        """
        downloads any file to s3.
        """
        with open(local_path, 'wb') as file:
            self.client.download_fileobj(bucket_name, key, file)
        return print(f"Downloaded file to: {local_path}")

    def download_zip(self, local_path, folder, type_, file_name):
        """
        downloads a zip file from s3.
        """
        print('Try to download from', self.bucket_name,
              f'{self.model_name}/{self.version}/{folder}/{type_}/{file_name}')
        with open(local_path, 'wb') as file:
            self.client.download_fileobj(self.bucket_name,
                                         f'{self.model_name}/{self.version}/'
                                         f'{folder}/{type_}/{file_name}', file)
        return print(f"Downloaded file to: {local_path}")

    def unzip(self, path_to_zip, extract_location=None):
        """
        unzips a zip file.
        """
        if extract_location is None:
            dirname = os.path.dirname(path_to_zip)
            extract_location = dirname
        with zipfile.ZipFile(path_to_zip, "r") as zip_ref:
            zip_ref.extractall(extract_location)
        return print(f"unzipped at: {extract_location}")

    def zip_and_upload(self, local_path, folder, type_, file_name):
        """
        zips a file then uploads to s3.
        """
        path = shutil.make_archive(local_path, 'zip', local_path)
        try:
            self.client.upload_file(path, self.bucket_name,
                                    f'{self.model_name}/{self.version}/'
                                    f'{folder}/{type_}/{file_name}')
            os.remove(path)
            print(f"\n***Locally saved {path} was successfully deleted.***\n")


        except ClientError as error:
            logging.error(error)
            return False
        return print(
            f'Uploaded Path: s3://{self.bucket_name}/{self.model_name}/{self.version}/'
            f'{folder}/{type_}/{file_name}')

    def pandas_dataframe_to_s3(self, input_datafame, folder, type_, file_name):
        """
        upload a pandas dataframe as parquet to s3.
        """
        out_buffer = BytesIO()
        input_datafame.to_parquet(out_buffer, index=False)

        try:
            self.client.put_object(Bucket=self.bucket_name,
                                   Key=f'{self.model_name}/{self.version}/{folder}'
                                       f'/{type_}/{file_name}',
                                   Body=out_buffer.getvalue())
        except ClientError as error:
            logging.error(error)
            return False
        return print(
            f'Uploaded Path: s3://{self.bucket_name}/{self.model_name}/{self.version}/{folder}/'
            f'{type_}/{file_name}')

    def write_tensor(self, tensor, folder, type_, file_name):
        """
        writes a tensor to s3.
        """
        bytes_ = BytesIO()
        np.save(bytes_, tensor, allow_pickle=True)
        bytes_.seek(0)
        try:
            self.client.upload_fileobj(Fileobj=bytes_, Bucket=self.bucket_name,
                                       Key=f'{self.model_name}/{self.version}/'
                                           f'{folder}/{type_}/{file_name}')
        except ClientError as error:
            logging.error(error)
            return False
        return print(f'Uploaded Path: s3://{self.bucket_name}/{self.model_name}/'
                     f'{self.version}/{folder}/{type_}/{file_name}')

    def awswrangler_pandas_dataframe_to_s3(self, input_datafame, folder, type_, file_name):
        """
        writes a pandas dataframe as parquet to s3.
        """
        wr.s3.to_parquet(input_datafame,
                         path=f"s3://{self.bucket_name}/{self.model_name}/"
                              f"{self.version}/{folder}/{type_}/{file_name}")
        print(
            f"uploaded to: s3://{self.bucket_name}/{self.model_name}/"
            f"{self.version}/{folder}/{type_}/{file_name}")

    def read_tensor(self, folder, type_, file_name):
        """
        reads a tensor from s3.
        """
        bytes_ = BytesIO()
        try:
            self.client.download_fileobj(Fileobj=bytes_, Bucket=self.bucket_name,
                                         Key=f'{self.model_name}/{self.version}/'
                                             f'{folder}/{type_}/{file_name}')
        except ClientError as error:
            logging.error(error)
            return False

        bytes_.seek(0)
        tensor = np.load(bytes_, allow_pickle=True)
        print(
            f"tensor read from: s3://{self.bucket_name}/{self.model_name}/"
            f"{self.version}/{folder}/{type_}/{file_name}")
        return tensor

    def upload_directory(self, local_path, folder, type_):
        """
        uploads all the contents of a dir to s3.
        """
        is_exist = os.path.exists(local_path)
        if is_exist is False:
            raise Exception(f"Local path does not exist: {local_path}")
        for root, _, files in os.walk(local_path):
            for file in files:
                try:
                    self.client.upload_file(os.path.join(root, file),
                                            Bucket=self.bucket_name,
                                            Key=f'{self.model_name}/{self.version}/'
                                                f'{folder}/{type_}/{file}')
                except ClientError as error:
                    logging.error(error)
                    return False
                print(
                    f'file written to: s3://{self.bucket_name}/{self.model_name}/'
                    f'{self.version}/{folder}/{type_}/{file}')
        return print("Commpleted")

    def pyspark_write_parquet(self, dataframe, folder, type_, ):
        """
        write pyspark dataframe as parquet to s3.
        """
        dataframe.write.mode('overwrite').parquet(
            f's3a://{self.bucket_name}/{self.model_name}/{self.version}/{folder}/{type_}/')
        return print(
            f'parquet written to: s3://{self.bucket_name}/{self.model_name}/'
            f'{self.version}/{folder}/{type_}/')

    def read_parquet_to_pandas_df(self, folder, type_, file_name):
        """
         reads a single parquet file as pandas df.
        """
        # Read the parquet file
        buffer = BytesIO()
        object_ = self.resource.Object(self.bucket_name,
                                       f'{self.model_name}/{self.version}/'
                                       f'{folder}/{type_}/{file_name}')
        object_.download_fileobj(buffer)
        dataframe = pd.read_parquet(buffer)
        return dataframe

    # Read single parquet file from S3
    def pd_read_s3_parquet(self, key, **args):
        obj = self.client.get_object(Bucket=self.bucket_name, Key=key)
        return pd.read_parquet(BytesIO(obj['Body'].read()), **args)

    # Read multiple parquets from a folder on S3 generated by spark
    def pd_read_s3_multiple_parquets(self, filepath, verbose=False, **args):
        if not filepath.endswith('/'):
            filepath = filepath + '/'  # Add '/' to the end
        s3_keys = [item.key for item in self.resource.Bucket(self.bucket_name).objects.filter(Prefix=filepath)
                   if item.key.endswith('.parquet')]
        if not s3_keys:
            print('No parquet found in', self.bucket_name, filepath)
        elif verbose:
            print('Load parquets:')
            for p in s3_keys:
                print(p)
        dfs = [self.pd_read_s3_parquet(self, **args) for key in s3_keys]  ## TODO modify input args
        return pd.concat(dfs, ignore_index=True)

