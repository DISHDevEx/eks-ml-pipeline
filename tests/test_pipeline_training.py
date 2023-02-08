"""
Module to contain all tests pertaining to pipelines that train models
"""
from eks_ml_pipeline import TrainTestPipelines
import boto3


def test_ae_pipeline_training(ae_train_input, bucket_name):
    """
    Verifies the training logic for the autoencoder training pipeline.
    Checks if a model is saved to the correct s3 path after training.
    Checks if the pipeline is able to read our presaved training tensor from s3.

     Inputs: fixtures of training input and bucket name.
     Output: None

     multiple try catch added to save time, train() can take 45seconds+
    """
    s3_client = boto3.client("s3")
    ttp_ae = TrainTestPipelines(ae_train_input)
    ttp_ae.train()

    errors = []
    try:
        s3_client.head_object(
            Bucket=bucket_name,
            Key="pytest_autoencoder_ad/v0.0.1/models/zipped_models/train_autoencoder_ad_model_v0.0.1_aeDummyDataTrain.zip",
        )
        s3_client.delete_object(
            Bucket=bucket_name,
            Key="pytest_autoencoder_ad/v0.0.1/models/zipped_models/train_autoencoder_ad_model_v0.0.1_aeDummyDataTrain.zip",
        )
    except Exception as error_message:
        errors.append(error_message)

    try:
        s3_client.head_object(
            Bucket=bucket_name,
            Key="pytest_autoencoder_ad/v0.0.1/models/onnx_models/train_autoencoder_ad_model_v0.0.1_aeDummyDataTrain.onnx",
        )
        s3_client.delete_object(
            Bucket=bucket_name,
            Key="pytest_autoencoder_ad/v0.0.1/models/onnx_models/train_autoencoder_ad_model_v0.0.1_aeDummyDataTrain.onnx",
        )
    except Exception as error_message:
        errors.append(error_message)

    assert len(errors) == 0, f"{errors}"


def test_pca_pipeline_training(pca_train_input, bucket_name):
    """
    Verifies the training logic for the pca training pipeline.
    Checks if a model is saved to the correct s3 path after training.
    Checks if the pipeline is able to read our presaved training tensor from s3.

     Inputs: fixtures of training input and bucket name
     Output: None

     multiple try catch added to save time, train() can take 45seconds+
    """
    s3_client = boto3.client("s3")
    ttp_pca = TrainTestPipelines(pca_train_input)
    ttp_pca.train()
    errors = []

    try:
        s3_client.head_object(
            Bucket=bucket_name,
            Key="pytest_pca_ad/v0.0.1/models/npy_models/train_pca_ad_model_v0.0.1_pcaDummyDataTrain.npy",
        )
        s3_client.delete_object(
            Bucket=bucket_name,
            Key="pytest_pca_ad/v0.0.1/models/npy_models/train_pca_ad_model_v0.0.1_pcaDummyDataTrain.npy",
        )
    except Exception as error_message:
        errors.append(error_message)

    assert len(errors) == 0, f"{errors}"
