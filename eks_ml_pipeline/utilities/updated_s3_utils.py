import numpy as np
import boto3
import os
import shutil
import zipfile
from io import BytesIO



class s3_utills:

    def __init__(self):

        self.bucket_name = ""
        self.model_name = ""
        self.version = ""
        self.file_name = ""
        self.extension = ""


    def boto3_client(self):
        client = boto3.client('s3')
        return client

    def read_tensor(self):
        print(f"reading tensor from: {self.bucket_name}/{self.model_name}/{self.version}/data/tensors/{self.file_name}.npy")
        client = boto3.client('s3')
        bytes_ = BytesIO()
        client.download_fileobj(Fileobj=bytes_, Bucket=self.bucket_name,
                                Key=f'{self.model_name}/{self.version}/data/tensors/{self.file_name}.npy')
        bytes_.seek(0)
        tensor = np.load(bytes_, allow_pickle=True)

        return tensor

    def write_tensor(self, tensor, bucket_name, model_name, version, flag, file_name):
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
                                  Key=f'{model_name}/{version}/models/{file_name}.npy')
            path = f'{bucket_name}/{model_name}/{version}/models/{file_name}.npy'

        print(f"writing tensor to: {bucket_name}/{model_name}/{version}/data/tensors/{file_name}.npy")

        return path

    def uploadDirectory(self, local_path, bucketname, model_name, version):
        for root, dirs, files in os.walk(local_path):
            for file in files:
                client = boto3.client('s3')
                client.upload_file(os.path.join(root, file), bucketname, model_name + '/' + version + '/models/' + file)
        print(f"file uploaded too: {bucketname}/{model_name}/{version}/{models}/{file}")

    def write_parquet(self, df, bucket_name, model_name, version, model_data_type):
        df.write.mode('overwrite').parquet(
            f's3a://{bucket_name}/{model_name}/{version}/data/pyspark/{model_data_type}/')

    def upload_zip(self, local_path, bucket_name, model_name, version, file_name):
        path = shutil.make_archive(local_path, 'zip', local_path)
        client = boto3.client('s3')
        client.upload_file(path, bucket_name, model_name + '/' + version + '/models/' + file_name + ".zip")
        print(f"Zip file uploaded: {bucket_name}/{model_name}/{version}/models/{file_name}.zip")

    def download_zip(self, download_path, bucket_name, model_name, version, file_name):
        with open(download_path, 'wb') as f:
            client = boto3.client('s3')
            client.download_fileobj(bucket_name, model_name + '/' + version + '/models/' + file_name + ".zip", f)
        print("zip file downloaded to : ")

    def unzip(self, path_to_zip, extract_location):

        with zipfile.ZipFile(path_to_zip, "r") as zip_ref:
            zip_ref.extractall(extract_location)
        print("file unzipped")

    def pandas_dataframe_to_s3(self, input_datafame, bucket_name, model_name, version, model_data_type):
        client = boto3.client('s3')
        out_buffer = BytesIO()
        input_datafame.to_parquet(out_buffer, index=False)
        client.put_object(Bucket=bucket_name, Key=f"{model_name}/{version}/data/pandas_df/{model_data_type}.parquet",
                          Body=out_buffer.getvalue())
        return print("sucess")

    def awswrangler_pandas_dataframe_to_s3(self, input_datafame, bucket_name, model_name, version, model_data_type):
        import awswrangler as wr
        wr.s3.to_parquet(input_datafame,
                         path=f"s3://{bucket_name}/{model_name}/{version}/data/pandas_df/{model_data_type}.parquet")
        return print("sucess")

    def write_onnx(self, local_path, bucket_name, model_name, version, file_name):
        client = boto3.client('s3')
        client.upload_file(local_path, bucket_name, model_name + '/' + version + '/models/' + file_name + ".onnx")
        print(f"Zip file uploaded: {bucket_name}/{model_name}/{version}/models/{file_name}.onnx")





# writing
def write_tensor(tensor, bucket_name, model_name, version, flag, file_name):
    pass

def upload_zip(local_path, bucket_name, model_name, version, file_name):
    pass

def write_onnx(local_path, bucket_name, model_name, version, file_name):
    pass

def pandas_dataframe_to_s3(input_datafame, bucket_name, model_name, version, model_data_type):
    pass








# reading
def read_tensor(bucket_name, model_name,version, file_name):
    pass

def download_zip(download_path, bucket_name, model_name, version, file_name):
    pass




#different
def unzip(path_to_zip, extract_location):
    pass

def uploadDirectory(local_path, bucketname, model_name, version):
    pass

def write_parquet(df, bucket_name, model_name, version, model_data_type):
    pass

def awswrangler_pandas_dataframe_to_s3(input_datafame, bucket_name, model_name, version, model_data_type):
    pass