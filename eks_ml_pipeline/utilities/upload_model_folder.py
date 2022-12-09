import boto3
import os


def uploadDirectory(path,bucketname,model_name,version):
    for root,dirs,files in os.walk(path):
        for file in files:
            #print(os.path.join(root,file))
            #print(bucketname)
            #print(file)
            client = boto3.client('s3')
            client.upload_file(os.path.join(root,file),bucketname, model_name+'/'+version+'/models/'+file)