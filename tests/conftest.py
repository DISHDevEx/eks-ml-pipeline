import pytest
from devex_sdk import get_features
from eks_ml_pipeline import AutoencoderModelDish5g
from eks_ml_pipeline import PcaModelDish5g
import os

# functions to mark slow tests and skip them.
def pytest_addoption(parser):
    parser.addoption(
        "--slow", action="store_true", default=False, help="run (slow) performance tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as a (potentially slow) performance test")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip_perf = pytest.mark.skip(reason="need --slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_perf)
@pytest.fixture(scope="module")           
def Bucket_Name():   
    Bucket_Name = os.environ.get("BUCKET_NAME_PYTEST")
    return Bucket_Name

@pytest.fixture(scope="module")           
def aeTrainInput(Bucket_Name):
    """
    This fixture creates inputs to train the desired model. It includes all of bucket versioning and model versioning needed 
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

    #*****************************************************#
    #********** data and model input parameters **********#
    #*****************************************************#

    #feature_selection
    feature_group_name = "pytest_autoencoder_ad"
    feature_input_version = "v0.0.1"

    # data_locations
    data_bucketname = Bucket_Name
    train_data_filename = "aeDummyDataTrain.npy"
    test_data_filename = "aeDummyDataTest.npy"

    # save_model_locations
    save_model_local_path = "../test_autoencoder"
    model_bucketname = Bucket_Name
    model_name = 'train_autoencoder_ad'
    model_version = "v0.0.1"

    #Define model filename and path
    model_filename = '_'.join(
        [model_name,
        "model",
        model_version,
        train_data_filename.split(".")[-2] # all preceeding extension
        ])

    #********************************************#
    #********** initialize model class **********#
    #********************************************#

    features_df = get_features(feature_group_name, feature_input_version)
    model_parameters = features_df["model_parameters"].iloc[0]

    #Initialize autoencoder model class with specific parameters
    encode_decode_model = AutoencoderModelDish5g(
        time_steps = model_parameters["time_steps"],
        batch_size = model_parameters["batch_size"],
        epochs=1
        )

    # File flags
    upload_zip = True
    upload_onnx = True
    upload_npy = False
    clean_local_folder = True

    return [encode_decode_model,
            [feature_group_name,feature_input_version], # feature_selection,
            [data_bucketname, train_data_filename, test_data_filename], # data_locations
            [save_model_local_path, model_bucketname, model_filename], # save_model_locations,
            [upload_zip, upload_onnx, upload_npy, clean_local_folder] # file_flags
           ]
            

@pytest.fixture(scope="module")  
def aeTestInput(Bucket_Name):
    """
    This fixture creates inputs to evaluate the desired model. It includes all of bucket versioning and model versioning needed 
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

    #*****************************************************#
    #********** data and model input parameters **********#
    #*****************************************************#

    #feature_selection
    feature_group_name = "pytest_autoencoder_ad"
    feature_input_version = "v0.0.1"

    # data_locations
    data_bucketname = Bucket_Name
    train_data_filename = "aeDummyDataTrain.npy"
    test_data_filename = "aeDummyDataTest.npy"

    # save_model_locations
    save_model_local_path = "../test_autoencoder"
    model_bucketname = Bucket_Name
    model_name = 'test_autoencoder_ad'
    model_version = "v0.0.1"

    #Define model filename and path
    model_filename = '_'.join(
        [model_name,
        "model",
        model_version,
        train_data_filename.split(".")[-2] # all preceeding extension
        ])

    #********************************************#
    #********** initialize model class **********#
    #********************************************#

    features_df = get_features(feature_group_name, feature_input_version)
    model_parameters = features_df["model_parameters"].iloc[0]

    #Initialize autoencoder model class with specific parameters
    encode_decode_model = AutoencoderModelDish5g(
        time_steps = model_parameters["time_steps"],
        batch_size = model_parameters["batch_size"],
        epochs=1
        )

    # File flags
    upload_zip = True
    upload_onnx = True
    upload_npy = False
    clean_local_folder = True

    return [encode_decode_model,
            [feature_group_name,feature_input_version], # feature_selection,
            [data_bucketname, train_data_filename, test_data_filename], # data_locations
            [save_model_local_path, model_bucketname, model_filename], # save_model_locations,
            [upload_zip, upload_onnx, upload_npy, clean_local_folder] # file_flags
           ]


@pytest.fixture(scope="module")           
def pcaTrainInput(Bucket_Name):
    """
    This fixture creates inputs to train the desired model. It includes all of bucket versioning and model versioning needed 
    as well as the file locations for a pipeline. 
    Parameters
    ----------
    None

    Returns
    -------
        List of parameters for container rec type
        required by pca model
        training pipeline
    """

    #*****************************************************#
    #********** data and model input parameters **********#
    #*****************************************************#

    ##generate pipeline input params for pca

    # feature_selection
    feature_group_name = "pytest_pca_ad"
    feature_input_version = "v0.0.1"

    # data_locations
    data_bucketname = Bucket_Name
    train_data_filename = "pcaDummyDataTrain.npy"
    test_data_filename = "pcaDummyDataTest.npy"

    # save_model_locations
    save_model_local_path = "../test_pca.npy"
    model_bucketname = Bucket_Name
    model_name = "train_pca_ad"
    model_version = "v0.0.1"
    #Define model filename
    model_filename = '_'.join(
        [model_name, "model",
         model_version,
         train_data_filename.split(".")[-2] # All preceeding extension.
        ])

    #********************************************#
    #********** initialize model class **********#
    #********************************************#

    features_df = get_features(feature_group_name, feature_input_version)
    features = features_df["feature_name"].to_list()
    model_parameters = features_df["model_parameters"].iloc[0]

    #Initialize pca model
    encode_decode_model = PcaModelDish5g(
        num_of_features = len(features),
        timesteps_per_slice = model_parameters["time_steps"]
        )

    # File flags
    upload_zip = False
    upload_onnx = False
    upload_npy = True
    clean_local_folder = True

    return [encode_decode_model,
            [feature_group_name,feature_input_version], # feature_selection,
            [data_bucketname, train_data_filename, test_data_filename], # data_locations
            [save_model_local_path, model_bucketname, model_filename], # save_model_locations,
            [upload_zip, upload_onnx, upload_npy, clean_local_folder] # file_flags
           ]


@pytest.fixture(scope="module")           
def pcaTestInput(Bucket_Name):
    """
    This fixture creates inputs to evaluate the desired model. It includes all of bucket versioning and model versioning needed 
    as well as the file locations for a pipeline. 
    Parameters
    ----------
    None

    Returns
    -------
        List of parameters for container rec type
        required by pca model
        training pipeline
    """

    #*****************************************************#
    #********** data and model input parameters **********#
    #*****************************************************#

    ##generate pipeline input params for pca

    # feature_selection
    feature_group_name = "pytest_pca_ad"
    feature_input_version = "v0.0.1"

    # data_locations
    data_bucketname = Bucket_Name
    train_data_filename = "pcaDummyDataTrain.npy"
    test_data_filename = "pcaDummyDataTest.npy"

    # save_model_locations
    save_model_local_path = "../test_pca.npy"
    model_bucketname = Bucket_Name
    model_name = "test_pca_ad"
    model_version = "v0.0.1"
    #Define model filename
    model_filename = '_'.join(
        [model_name, "model",
         model_version,
         train_data_filename.split(".")[-2] # All preceeding extension.
        ])

    #********************************************#
    #********** initialize model class **********#
    #********************************************#

    features_df = get_features(feature_group_name, feature_input_version)
    features = features_df["feature_name"].to_list()
    model_parameters = features_df["model_parameters"].iloc[0]

    #Initialize pca model
    encode_decode_model = PcaModelDish5g(
        num_of_features = len(features),
        timesteps_per_slice = model_parameters["time_steps"]
        )

    # File flags
    upload_zip = False
    upload_onnx = False
    upload_npy = True
    clean_local_folder = True

    return [encode_decode_model,
            [feature_group_name,feature_input_version], # feature_selection,
            [data_bucketname, train_data_filename, test_data_filename], # data_locations
            [save_model_local_path, model_bucketname, model_filename], # save_model_locations,
            [upload_zip, upload_onnx, upload_npy, clean_local_folder] # file_flags
           ]
