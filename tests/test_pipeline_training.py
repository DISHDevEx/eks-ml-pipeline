from eks_ml_pipeline import TrainTestPipelines
import boto3


def test_aePipelineTraining(aeTrainInput,Bucket_Name):
    """
    This testing module verifies the training logic for the autoencoder training pipeline. It checks if a model is saved to the correct s3 path after training.
    And it also checks if the pipeline is able to read our presaved training tensor from s3. 

     Inputs: fixtures of training input and bucket name. 
     Output: None
    """
    s3 = boto3.client('s3')
    ttp_ae = TrainTestPipelines(aeTrainInput)
    ttp_ae.train()
    
    errors = []
    try:
        zip_file_head = s3.head_object(Bucket=Bucket_Name, Key='pytest_autoencoder_ad/v0.0.1/models/zipped_models/train_autoencoder_ad_model_v0.0.1_aeDummyDataTrain.zip')
        s3.delete_object(Bucket=Bucket_Name, Key='pytest_autoencoder_ad/v0.0.1/models/zipped_models/train_autoencoder_ad_model_v0.0.1_aeDummyDataTrain.zip')
    except Exception as e:
        errors.append(e)
        
    try:   
        onnx_file_head = s3.head_object(Bucket=Bucket_Name, Key='pytest_autoencoder_ad/v0.0.1/models/onnx_models/train_autoencoder_ad_model_v0.0.1_aeDummyDataTrain.onnx')
        s3.delete_object(Bucket=Bucket_Name, Key='pytest_autoencoder_ad/v0.0.1/models/onnx_models/train_autoencoder_ad_model_v0.0.1_aeDummyDataTrain.onnx')
    except Exception as e:
        errors.append(e)
        
    
        
    assert(len(errors)==0)
    
    
    
def test_pcaPipelineTraining(pcaTrainInput,Bucket_Name):
    """
    This testing module verifies the training logic for the pca training pipeline. It checks if a model is saved to the correct s3 path after training.
    And it also checks if the pipeline is able to read our presaved training tensor from s3. 

     Inputs: fixtures of training input and bucket name 
     Output: None
    """
    s3 = boto3.client('s3')
    ttp_pca = TrainTestPipelines(pcaTrainInput)
    ttp_pca.train()
    errors = []
    
    try:
        zip_file_head = s3.head_object(Bucket=Bucket_Name, Key='pytest_pca_ad/v0.0.1/models/npy_models/train_pca_ad_model_v0.0.1_pcaDummyDataTrain.npy')
        s3.delete_object(Bucket=Bucket_Name, Key='pytest_pca_ad/v0.0.1/models/npy_models/train_pca_ad_model_v0.0.1_pcaDummyDataTrain.npy')
    except Exception as e:
        errors.append(e)
        
    assert(len(errors)==0)

    
