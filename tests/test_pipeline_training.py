from eks_ml_pipeline import TrainTestPipelines
import boto3

def test_aePipelineTraining(aeTrainInput):
    s3 = boto3.client('s3')
    ttp_ae = TrainTestPipelines(aeTrainInput)
    ttp_ae.train()
    
    errors = []
    
    try:
        zip_file_head = s3.head_object(Bucket='dish-5g.core.pd.g.dp.eks.logs.e', Key='pytest_autoencoder_ad/v0.0.1/models/zipped_models/train_autoencoder_ad_model_v0.0.1_aeDummyDataTrain.zip')
        s3.delete_object(Bucket='dish-5g.core.pd.g.dp.eks.logs.e', Key='pytest_autoencoder_ad/v0.0.1/models/zipped_models/train_autoencoder_ad_model_v0.0.1_aeDummyDataTrain.zip')
    except Exception as e:
        errors.append(e)
        
    try:   
        onnx_file_head = s3.head_object(Bucket='dish-5g.core.pd.g.dp.eks.logs.e', Key='pytest_autoencoder_ad/v0.0.1/models/onnx_models/train_autoencoder_ad_model_v0.0.1_aeDummyDataTrain.onnx')
        s3.delete_object(Bucket='dish-5g.core.pd.g.dp.eks.logs.e', Key='pytest_autoencoder_ad/v0.0.1/models/onnx_models/train_autoencoder_ad_model_v0.0.1_aeDummyDataTrain.onnx')
    except Exception as e:
        errors.append(e)
        
    
        
    assert(len(errors)==0)
    
    
    
def test_pcaPipelineTraining(pcaTrainInput):
    s3 = boto3.client('s3')
    ttp_pca = TrainTestPipelines(pcaTrainInput)
    ttp_pca.train()
    
    errors = []
    
    try:
        zip_file_head = s3.head_object(Bucket='dish-5g.core.pd.g.dp.eks.logs.e', Key='pytest_pca_ad/v0.0.1/models/npy_models/train_pca_ad_model_v0.0.1_pcaDummyDataTrain.npy')
        s3.delete_object(Bucket='dish-5g.core.pd.g.dp.eks.logs.e', Key='pytest_pca_ad/v0.0.1/models/npy_models/train_pca_ad_model_v0.0.1_pcaDummyDataTrain.npy')
    except Exception as e:
        errors.append(e)
        
    assert(len(errors)==0)

    
