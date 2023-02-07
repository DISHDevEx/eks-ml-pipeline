from eks_ml_pipeline import TrainTestPipelines
import boto3
import os


def test_ae_pipeline_eval(ae_test_input, bucket_name):
    """
    This testing module verifies the evaluation logic for the autoencoder evaluation pipeline. It checks if a model is read from the correct s3 path after training.
    And it also checks if the pipeline is able to read our presaved evaluation tensor from s3. 
    Then the test checks if the residuals and predictions are stored in the correct s3 path. 
    
    Inputs: fixtures of training input and bucket name. 
    Output: None
    """
    s3 = boto3.client("s3")
    ttp_ae = TrainTestPipelines(ae_test_input)
    ttp_ae.test()

    errors = []

    try:
        pred_key = "pytest_autoencoder_ad/v0.0.1/models/predictions/aeDummyDataTest_predictions.npy"
        preds_file_head = s3.head_object(Bucket=bucket_name, Key=pred_key)
        s3.delete_object(Bucket=bucket_name, Key=pred_key)
    except Exception as e:
        errors.append(e)

    try:
        res_file_head = s3.head_object(
            Bucket=bucket_name,
            Key="pytest_autoencoder_ad/v0.0.1/models/predictions/aeDummyDataTest_residuals.npy",
        )
        s3.delete_object(
            Bucket=bucket_name,
            Key="pytest_autoencoder_ad/v0.0.1/models/predictions/aeDummyDataTest_residuals.npy",
        )
    except Exception as e:
        errors.append(e)

    if os.path.exists("../test_autoencoder") == True:
        errors.append("AE Model not deleted correctly")

    assert len(errors) == 0


def test_pca_pipeline_eval(pca_test_input, bucket_name):
    """
    This testing module verifies the evaluation logic for the pca evaluation pipeline. It checks if a model is read from the correct s3 path after training.
    And it also checks if the pipeline is able to read our presaved evaluation tensor from s3. 
    Then the test checks if the residuals and predictions are stored in the correct s3 path. 

     Inputs: fixtures of training input and bucket name. 
     Output: None
    """
    s3 = boto3.client("s3")
    ttp_pca = TrainTestPipelines(pca_test_input)
    ttp_pca.test()
    errors = []

    try:
        preds_file_head = s3.head_object(
            Bucket=bucket_name,
            Key="pytest_pca_ad/v0.0.1/models/predictions/pcaDummyDataTest_predictions.npy",
        )
        s3.delete_object(
            Bucket=bucket_name,
            Key="pytest_pca_ad/v0.0.1/models/predictions/pcaDummyDataTest_predictions.npy",
        )
    except Exception as e:
        errors.append(e)

    try:
        res_file_head = s3.head_object(
            Bucket=bucket_name,
            Key="pytest_pca_ad/v0.0.1/models/predictions/pcaDummyDataTest_residuals.npy",
        )
        s3.delete_object(
            Bucket=bucket_name,
            Key="pytest_pca_ad/v0.0.1/models/predictions/pcaDummyDataTest_residuals.npy",
        )
    except Exception as e:
        errors.append(e)

    if os.path.exists("../test_pca.npy") == True:
        errors.append("AE Model not deleted correctly")

    assert len(errors) == 0
