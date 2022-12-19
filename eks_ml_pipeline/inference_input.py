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
    
    raw_data_s3_path = 's3://dish-5g.core.pd.g.dp.eks.logs.e/inference_data/Node/Node_2022_9_11_12.parquet'
    sampling_column = "InstanceId"
    file_prefix = 'inference'
    
    return [raw_data_s3_path, sampling_column, file_prefix]

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