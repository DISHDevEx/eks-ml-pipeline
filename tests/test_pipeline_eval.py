"""
Module to contain all tests pertaining to pipelines that evaluate models
"""
import os
from eks_ml_pipeline import TrainTestPipelines
import boto3

def test_ae_pipeline_eval(ae_test_input, bucket_name):
    """
    Verify the evaluation logic for the autoencoder evaluation pipeline.
    Check if a model is read from the correct s3 path after training.
    Check if the pipeline is able to read our presaved evaluation tensor from s3.
    Checks if the residuals and predictions are stored in the correct s3 path.

    Inputs: fixtures of training input and bucket name.
    Output: None

    multiple try catch added to save time, test() can take 45seconds+
    """
    s3_client = boto3.client("s3")
    ttp_ae = TrainTestPipelines(ae_test_input)
    ttp_ae.test()

    errors = []

    try:
        s3_client.head_object(
            Bucket=bucket_name,
            Key="pytest_autoencoder_ad/v0.0.1/models/predictions/test_autoencoder_ad_model_v0.0.1_aeDummyDataTrain_aeDummyDataTest_predictions.npy",
        )
        s3_client.delete_object(
            Bucket=bucket_name,
            Key="pytest_autoencoder_ad/v0.0.1/models/predictions/test_autoencoder_ad_model_v0.0.1_aeDummyDataTrain_aeDummyDataTest_predictions.npy",
        )
    except Exception as error_message:
        errors.append(error_message)

    try:
        s3_client.head_object(
            Bucket=bucket_name,
            Key="pytest_autoencoder_ad/v0.0.1/models/predictions/test_autoencoder_ad_model_v0.0.1_aeDummyDataTrain_aeDummyDataTest_residuals.npy",
        )
        s3_client.delete_object(
            Bucket=bucket_name,
            Key="pytest_autoencoder_ad/v0.0.1/models/predictions/test_autoencoder_ad_model_v0.0.1_aeDummyDataTrain_aeDummyDataTest_residuals.npy",
        )
    except Exception as error_message:
        errors.append(error_message)

    if os.path.exists("../test_autoencoder"):
        errors.append("AE Model not deleted correctly")

    assert len(errors) == 0, f"{errors}"


def test_pca_pipeline_eval(pca_test_input, bucket_name):
    """
    Verify the evaluation logic for the pca evaluation pipeline.
    Checks if a model is read from the correct s3 path after training.
    Checks if the pipeline is able to read our presaved evaluation tensor from s3.
    Checks if the residuals and predictions are stored in the correct s3 path.

    Inputs: fixtures of training input and bucket name.
    Output: None

    multiple try catch added to save time, test() can take 45seconds+
    """
    s3_client = boto3.client("s3")
    ttp_pca = TrainTestPipelines(pca_test_input)
    ttp_pca.test()
    errors = []

    try:
        s3_client.head_object(
            Bucket=bucket_name,
            Key="pytest_pca_ad/v0.0.1/models/predictions/test_pca_ad_model_v0.0.1_pcaDummyDataTrain_pcaDummyDataTest_predictions.npy",
        )
        s3_client.delete_object(
            Bucket=bucket_name,
            Key="pytest_pca_ad/v0.0.1/models/predictions/test_pca_ad_model_v0.0.1_pcaDummyDataTrain_pcaDummyDataTest_predictions.npy",
        )
    except Exception as error_message:
        errors.append(error_message)

    try:
        s3_client.head_object(
            Bucket=bucket_name,
            Key="pytest_pca_ad/v0.0.1/models/predictions/test_pca_ad_model_v0.0.1_pcaDummyDataTrain_pcaDummyDataTest_residuals.npy",
        )
        s3_client.delete_object(
            Bucket=bucket_name,
            Key="pytest_pca_ad/v0.0.1/models/predictions/test_pca_ad_model_v0.0.1_pcaDummyDataTrain_pcaDummyDataTest_residuals.npy",
        )
    except Exception as error_message:
        errors.append(error_message)

    if os.path.exists("../test_pca.npy"):
        errors.append("AE Model not deleted correctly")

    assert len(errors) == 0, f"{errors}"
