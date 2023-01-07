"""
Contributed by Evgeniya Dontsova
MSS Dish 5g - Pattern Detection

these are the parameters for inference pipeline

"""

def node_inference_input():
    
    """
    outputs
    -------
            list of parameters for node rec type
            required for inference pipeline
            
    """
    
    rec_type = 'Node'
    
    ##feature engineering specs
    sampling_column = "InstanceId"
    
    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "9"
    partition_day = "11"
    partition_hour = "1"
    spark_config_setup = "384gb"
    
    ##s3 bucket for raw inference data
    data_bucketname = "dish-5g.core.pd.g.dp.eks.logs.e"
    
    ##full s3 model path
    model_s3_path = ""
    
    #raw_data_s3_path = 's3://dish-5g.core.pd.g.dp.eks.logs.e/inference_data/Node/Node_2022_9_11_12.parquet'
    #sampling_column = "InstanceId"
    #file_prefix = 'inference'
    
    return [rec_type, sampling_column,
            partition_year, partition_month, partition_day, partition_hour, 
            spark_config_setup, data_bucketname, model_s3_path]

def pod_inference_input():
    
    """
    outputs
    -------
            list of parameters for pod rec type
            required for inference pipeline
            
    """
    
    raw_data_s3_path = 's3://dish-5g.core.pd.g.dp.eks.logs.e/inference_data/Pod/Pod_2022_7_10_20.parquet'
    sampling_column = ("kubernetes", "pod_id")
    file_prefix = 'inference'
    
    return [raw_data_s3_path, sampling_column, file_prefix]

def container_inference_input():
    
    """
    outputs
    -------
            list of parameters for container rec type
            required for inference pipeline
            
    """
    
    raw_data_s3_path = 's3://dish-5g.core.pd.g.dp.eks.logs.e/inference_data/Container/Container_2022_8_20_9.parquet'
    sampling_column = "InstanceId"
    file_prefix = 'inference'
    
    return [raw_data_s3_path, sampling_column, file_prefix]