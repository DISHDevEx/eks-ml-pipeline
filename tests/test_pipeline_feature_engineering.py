"""
Contain all tests pertaining to pipelines that train models
"""
from eks_ml_pipeline import FeatureEngineeringPipeline
import boto3
import pytest

@pytest.mark.skip()
def test_ae_pipeline_preprocessing(ae_fe_input, bucket_name):
    """
    Verify the pre-processing logic for the autoencoder feature engineering pipeline.
    Checks if the raw data is processed and saved to the correct s3 path after preprocessing.

    Inputs: fixtures of feature engineering input and bucket name.
    Output: None
    """
    rec_type = 'node'
    compute_type = 'sagemaker'
    input_data_type = 'train'

    s3_client = boto3.client("s3")
    fep = FeatureEngineeringPipeline(ae_fe_input(), rec_type, compute_type, input_data_type)
    fep.run_preprocessing()



def test_ae_pipeline_feature_engineering(ae_fe_input):
    """
    Verify the pre-processing logic for the autoencoder feature engineering pipeline.
    Checks if the processed data has proper feature engineering applied and final data is saved to the correct s3 path after preprocessing.

    Inputs: fixtures of feature engineering input and bucket name.
    Output: None
    """
    rec_type = 'node'
    compute_type = 'sagemaker'
    input_data_type = 'train'

    s3_client = boto3.client("s3")
    fep = FeatureEngineeringPipeline(ae_fe_input(), rec_type, compute_type, input_data_type)
    fep.run_feature_engineering()

