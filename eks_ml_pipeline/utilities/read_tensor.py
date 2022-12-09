from io import BytesIO
import numpy as np
from urllib.parse import urlparse
import boto3




def read_tensor(bucket,filename):
    print(f"reading tensor from: {bucket+filename}")
    client = boto3.client('s3')
    bytes_ = BytesIO()
    client.download_fileobj(Fileobj=bytes_, Bucket=bucket, Key=filename)
    bytes_.seek(0)
    tensor = np.load(bytes_, allow_pickle=True)
    
    return tensor