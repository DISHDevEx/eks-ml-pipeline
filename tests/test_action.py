import os
import pytest
import sys
import json
from glob import glob
import tempfile
import shutil
import numpy as np
import pandas as pd
from devex_sdk import get_features, EKS_Connector, Spark_Utils
from eks_ml_pipeline import AutoencoderModelDish5g
from eks_ml_pipeline import S3Utilities



@pytest.fixture(scope="module")
def bucket_name():
    """
    Get bucket name from the github workflow runner secrets
    """
    BUCKET_NAME = os.environ.get("BUCKET_NAME_PYTEST") 
    return BUCKET_NAME


def test_ae_train_input(bucket_name):
    """
    Create inputs to train the desired model.
    Includes all of bucket versioning and model versioning needed
    as well as the file locations for a pipeline.
    Parameters
    ----------
    None

    Returns
    -------
    training_inputs
        List of parameters for node rec type
        required by autoencoder model
        training pipeline
    """

    # *****************************************************#
    # ********** data and model input parameters **********#
    # *****************************************************#

    # feature_selection
    feature_group_name = "pytest_autoencoder_ad"
    feature_input_version = "v0.0.1"

    # data_locations
    data_bucketname = bucket_name
    train_data_filename = "aeDummyDataTrain.npy"
    test_data_filename = "aeDummyDataTest.npy"

    # save_model_locations
    save_model_local_path = "../test_autoencoder"
    model_bucketname = bucket_name
    model_name = "train_autoencoder_ad"
    model_version = "v0.0.0Pytest"

    # Define model filename and path
    model_filename = "_".join(
        [
            model_name,
            "model",
            model_version,
            train_data_filename.split(".")[-2],  # all preceeding extension
        ]
    )
    
    print('Model Filename:', model_filename)

    # ********************************************#
    # ********** initialize model class **********#
    # ********************************************#

    #features_df = get_features(feature_group_name, feature_input_version)
    
    all_features_path = glob(os.path.join(os.path.dirname(__file__), "eks_feature_store", "*.json"))
    
    print(all_features_path)

    for count,file_name in enumerate(all_features_path):
        with open(file_name) as f:
            feature_data = json.load(f)
            if count == 0 :
                features_df = pd.json_normalize(data=feature_data, record_path='features_list', 
                            meta=['feature_group_name', 'feature_group_description', 'model_type', 'problem_type', 'created_by', 'version', 'model_parameters'])
            else:
                features_df =  features_df.append(pd.json_normalize(data=feature_data, record_path='features_list', 
                            meta=['feature_group_name', 'feature_group_description', 'model_type', 'problem_type',  'created_by', 'version', 'model_parameters']))

    if feature_group_name != "" and feature_input_version != "":
        features_df = features_df[(features_df['feature_group_name'] == feature_group_name) & (features_df['version'] == feature_input_version)]
    
    
    
    print(features_df.columns)
    print(features_df.count())
    
    print (features_df["model_parameters"].iloc[0])
    
    assert len(features_df.columns) == 11