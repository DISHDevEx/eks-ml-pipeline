import numpy as np
import boto3
import os
import zipfile
from io import BytesIO
import shutil
import awswrangler as wr
import logging
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
            
            writing_path: STRING
            path to write files locally
            
            folder: STRING
            folder name to differentiate betwwen models or data etc
            
            type_: STRING
            type name to differentiate betwwen tensors or pandas_parquet etc
            
            file_name: STRING
            name of the file with extention
            

    outputs
    -------
            upload_file(): path
            download_file: path
            download_zip: path
            unzip: path
            zip_and_upload: path
            pandas_dataframe_to_s3: path
            write_tensor: path
            awswrangler_pandas_dataframe_to_s3: path
            read_tensor(): variable: numpy tensor
            upload_directory: path
            pyspark_write_parquet: path
    """

    def __init__(self,
                 bucket_name=None,
                 model_name=None,
                 version=None):

        self.bucket_name = bucket_name
        self.model_name = model_name
        self.version = version
        self.client = boto3.client('s3')
        if self.bucket_name == None or self.model_name == None or self.version == None:
            raise Exception("Please Initialize Class parameters")


    def upload_file(self, local_path, bucket_name, key):

        #self.client.upload_file(local_path, bucket_name, key)
        try:
            response = self.client.upload_file(local_path, bucket_name, key)
        except ClientError as e:
            logging.error(e)
            return False
        print(f'Uploaded Path: s3://{bucket_name}/{key}')
        
    
    def download_file(self, local_path, bucket_name, key):
        with open(local_path, 'wb') as f:
            self.client.download_fileobj(bucket_name, key, f)
        print(f"Downloaded file to: {local_path}")
    
    def download_zip(self, writing_path, folder, type_, file_name):
        with open(writing_path, 'wb') as f:
            self.client.download_fileobj(self.bucket_name,
                                         f'{self.model_name}/{self.version}/{folder}/{type_}/{file_name}', f)
        print(f"Downloaded file to: {writing_path}")
    
    def unzip(self, path_to_zip, extract_location):

        with zipfile.ZipFile(path_to_zip, "r") as zip_ref:
            zip_ref.extractall(extract_location)
        print(f"unzipped at: {extract_location}")
    
    def zip_and_upload(self, local_path, folder, type_, file_name):

        path = shutil.make_archive(local_path, 'zip', local_path)
        try:
            response = self.client.upload_file(path, self.bucket_name, f'{self.model_name}/{self.version}/{folder}/{type_}/{file_name}')
        except ClientError as e:
            logging.error(e)
            return False
        print(f'Uploaded Path: s3://{self.bucket_name}/{self.model_name}/{self.version}/{folder}/{type_}/{file_name}')
        
    
    def pandas_dataframe_to_s3(self, input_datafame, folder, type_, file_name):

        out_buffer = BytesIO()
        input_datafame.to_parquet(out_buffer, index=False)
        
        
        try:
            response = self.client.put_object(Bucket=self.bucket_name,
                               Key=f'{self.model_name}/{self.version}/{folder}/{type_}/{file_name}',
                               Body=out_buffer.getvalue())
        except ClientError as e:
            logging.error(e)
            return False
        print(f'Uploaded Path: s3://{self.bucket_name}/{self.model_name}/{self.version}/{folder}/{type_}/{file_name}')
        
       

    def write_tensor(self, tensor, folder, type_, file_name):

        bytes_ = BytesIO()
        np.save(bytes_, tensor, allow_pickle=True)
        bytes_.seek(0)
        try:
            response = self.client.upload_fileobj(Fileobj=bytes_, Bucket=self.bucket_name,
                                   Key=f'{self.model_name}/{self.version}/{folder}/{type_}/{file_name}')
        except ClientError as e:
            logging.error(e)
            return False
        print(f'Uploaded Path: s3://{self.bucket_name}/{self.model_name}/{self.version}/{folder}/{type_}/{file_name}')
        
    
    def awswrangler_pandas_dataframe_to_s3(self, input_datafame,
                                           folder, type_, file_name):

        wr.s3.to_parquet(input_datafame,
                         path=f"s3://{self.bucket_name}/{self.model_name}/{self.version}/{folder}/{type_}/{file_name}")
        print(f"uploaded to: s3://{self.bucket_name}/{self.model_name}/{self.version}/{folder}/{type_}/{file_name}")

    def read_tensor(self, folder, type_, file_name):
        bytes_ = BytesIO()
        
        try:
            response = self.client.download_fileobj(Fileobj=bytes_, Bucket=self.bucket_name,
                                     Key=f'{self.model_name}/{self.version}/{folder}/{type_}/{file_name}')
        except ClientError as e:
            logging.error(e)
            return False
        
        bytes_.seek(0)
        tensor = np.load(bytes_, allow_pickle=True)
        print(f"tensor read from: s3://{self.bucket_name}/{self.model_name}/{self.version}/{folder}/{type_}/{file_name}")
        return tensor



    def upload_directory(self, local_path, folder, type_):

        for root, dirs, files in os.walk(local_path):
            for file in files:
                try:
                    response = self.client.upload_file(os.path.join(root, file), Bucket=self.bucket_name,
                                        Key=f'{self.model_name}/{self.version}/{folder}/{type_}/{file}')
                except ClientError as e:
                    logging.error(e)
                    return False
                
            print(f'file written to: s3://{self.bucket_name}/{self.model_name}/{self.version}/{folder}/{type_}/{file}')
    
    
    
    def pyspark_write_parquet(self, df,folder, type_):

        df.write.mode('overwrite').parquet(
            f's3a://{self.bucket_name}/{self.model_name}/{self.version}/{folder}/{type_}/')
        print(f'parquet written to: s3://{self.bucket_name}/{self.model_name}/{self.version}/{folder}/{type_}/')
    
    