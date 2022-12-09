from io import BytesIO
import numpy as np
from urllib.parse import urlparse
import boto3





def write_tensor(tensor,bucket_name,model_name,version,filename):
    client = boto3.client('s3')
    bytes_ = BytesIO()
    np.save(bytes_, tensor, allow_pickle=True)
    bytes_.seek(0)
    #client.put_object(Body=a, Bucket=bucket, Key='array.npy')
    client.upload_fileobj(Fileobj=bytes_, Bucket=bucket_name, Key=model_name+'/'+version+'/data/'+model_name+'_'+filename)
    print(f"Bucket Name: {bucket_name}")
    print(f"Key: {model_name+'/'+version+'/data/'+model_name+'_'+filename}")
    path = bucket_name+'/'+model_name+'/'+version+'/data/'+model_name+'_'+filename
    return path