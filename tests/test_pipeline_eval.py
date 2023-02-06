from eks_ml_pipeline import TrainTestPipelines
import boto3
import os

def test_aePipelineEval(aeTestInput,Bucket_Name):
    s3 = boto3.client('s3')
    ttp_ae = TrainTestPipelines(aeTestInput)
    ttp_ae.test()
    
    errors = []

    try:
        preds_file_head = s3.head_object(Bucket=Bucket_Name, Key='pytest_autoencoder_ad/v0.0.1/models/predictions/aeDummyDataTest_predictions.npy')
        s3.delete_object(Bucket=Bucket_Name, Key='pytest_autoencoder_ad/v0.0.1/models/predictions/aeDummyDataTest_predictions.npy')
    except Exception as e:
        errors.append(e)
        
    try:   
        res_file_head = s3.head_object(Bucket=Bucket_Name, Key='pytest_autoencoder_ad/v0.0.1/models/predictions/aeDummyDataTest_residuals.npy')
        s3.delete_object(Bucket=Bucket_Name, Key='pytest_autoencoder_ad/v0.0.1/models/predictions/aeDummyDataTest_residuals.npy')
    except Exception as e:
        errors.append(e)
        
    if(os.path.exists("../test_autoencoder") == True):
        errors.append("AE Model not deleted correctly")
    
        
    assert(len(errors)==0)
    
    
    
def test_pcaPipelineEval(pcaTestInput,Bucket_Name):
    s3 = boto3.client('s3')
    ttp_pca = TrainTestPipelines(pcaTestInput)
    ttp_pca.test()
    errors = []
    
    try:
        preds_file_head = s3.head_object(Bucket=Bucket_Name, Key='pytest_pca_ad/v0.0.1/models/predictions/pcaDummyDataTest_predictions.npy')
        s3.delete_object(Bucket='dish-5g.core.pd.g.dp.eks.logs.e', Key='pytest_pca_ad/v0.0.1/models/predictions/pcaDummyDataTest_predictions.npy')
    except Exception as e:
        errors.append(e)
        
    try:   
        res_file_head = s3.head_object(Bucket= Bucket_Name, Key='pytest_pca_ad/v0.0.1/models/predictions/pcaDummyDataTest_residuals.npy')
        s3.delete_object(Bucket= Bucket_Name, Key='pytest_pca_ad/v0.0.1/models/predictions/pcaDummyDataTest_residuals.npy')
    except Exception as e:
        errors.append(e)
        
    if(os.path.exists("../test_pca.npy") == True):
        errors.append("AE Model not deleted correctly")
        
        
    assert(len(errors)==0)



