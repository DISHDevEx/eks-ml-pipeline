import os
from dotenv import load_dotenv
from eks_ml_pipeline import FeatureEngineeringPipeline
from eks_ml_pipeline import node_autoencoder_fe_input



if __name__ == "__main__":
    load_dotenv()
    rec_type = 'Node'
    compute_type = 'sagemaker'
    aggregation_column = 'InstanceId'
    input_data_type = 'test'

    dpfep = FeatureEngineeringPipeline(node_autoencoder_fe_input(), aggregation_column, rec_type, compute_type, input_data_type)

    dpfep.run_preproceesing()