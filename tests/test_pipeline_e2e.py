from eks_ml_pipeline import TrainTestPipelines
import boto3

def test_aePipelineTraining(aeTestInput):
    s3 = boto3.client('s3')
    ttp_ae = TrainTestPipelines(aeTestInput)
    ttp_ae.train()
    
    errors = []
    
    try:
        zip_file_head = s3.head_object(Bucket='dish-5g.core.pd.g.dp.eks.logs.e', Key='test_autoencoder_ad/v0.0.1/models/zipped_models/test_autoencoder_ad_model_v0.0.1-test_aeDummyDataTrain.zip')
    except Exception as e:
        errors.append(e)
        
    try:   
        onnx_file_head = s3.head_object(Bucket='dish-5g.core.pd.g.dp.eks.logs.e', Key='test_autoencoder_ad/v0.0.1/models/onnx_models/test_autoencoder_ad_model_v0.0.1-test_aeDummyDataTrain.onna')
    except Exception as e:
        errors.append(e)
        
    assert(len(errors)==0)
    
    
    
def test_pcaPipelineTraining(aeTestInput):
    s3 = boto3.client('s3')
    ttp_ae = TrainTestPipelines(aeTestInput)
    ttp_ae.train()
    
    errors = []
    
    try:
        zip_file_head = s3.head_object(Bucket='dish-5g.core.pd.g.dp.eks.logs.e', Key='test_autoencoder_ad/v0.0.1/models/zipped_models/test_autoencoder_ad_model_v0.0.1-test_aeDummyDataTrain.zip')
    except Exception as e:
        errors.append(e)
        
    try:   
        onnx_file_head = s3.head_object(Bucket='dish-5g.core.pd.g.dp.eks.logs.e', Key='test_autoencoder_ad/v0.0.1/models/onnx_models/test_autoencoder_ad_model_v0.0.1-test_aeDummyDataTrain.onnx')
    except Exception as e:
        errors.append(e)
        
    assert(len(errors)==0)

    
# def test_pcapipelines():
#     ttp_pca = TrainTestPipelines(test_pca_input)