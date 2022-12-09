def write_parquet(df,bucket_name,model_name,version,model_data_type):
    df.write.mode('overwrite').parquet(f's3a://{bucket_name}/{model_name}/{version}/data/{model_data_type}/')