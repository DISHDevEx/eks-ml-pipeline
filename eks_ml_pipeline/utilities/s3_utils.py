from io import BytesIO
import numpy as np
from urllib.parse import urlparse
import boto3
import os


def read_tensor(bucket_name,filename):
    """
    inputs
    ------
            bucket_name: STRING
            s3 bucket name to read tensor from
            
            filename: STRING
            file name to read file from s3 
            
    outputs
    -------
            tensor : numpy tensor
            
    """
    print(f"reading tensor from: {bucket_name+filename}")
    client = boto3.client('s3')
    bytes_ = BytesIO()
    client.download_fileobj(Fileobj=bytes_, Bucket=bucket_name, Key=filename)
    bytes_.seek(0)
    tensor = np.load(bytes_, allow_pickle=True)
    
    return tensor




def write_tensor(tensor,bucket_name,model_name,version,filename):
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
            
            filename: STRING
            file name is userd to write the tensor with this file name
            
    outputs
    -------
            path : string
            path where will the tensor is stored in s3
            
    """
    client = boto3.client('s3')
    bytes_ = BytesIO()
    np.save(bytes_, tensor, allow_pickle=True)
    bytes_.seek(0)
    #client.put_object(Body=a, Bucket=bucket, Key='array.npy')
    client.upload_fileobj(Fileobj=bytes_, Bucket=bucket_name, Key=model_name+'/'+version+'/data/'+model_name+'_'+filename)
    print(f"Bucket Name: {bucket_name}")
    path = bucket_name+'/'+model_name+'/'+version+'/data/'+model_name+'_'+filename
    return path


        
def uploadDirectory(local_path,bucketname,model_name,version):
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
    for root,dirs,files in os.walk(local_path):
        for file in files:
            client = boto3.client('s3')
            client.upload_file(os.path.join(root,file),bucketname, model_name+'/'+version+'/models/'+file)

    
def write_parquet(df,bucket_name,model_name,version,model_data_type):
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
    df.write.mode('overwrite').parquet(f's3a://{bucket_name}/{model_name}/{version}/data/{model_data_type}/')