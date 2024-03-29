"""
Define fixtures and configuration
that can be reused throughout pytest without redefinition.
It also defines the configurations for pytest.
"""
import os
import pytest
from devex_sdk import get_features, EKS_Connector, Spark_Utils
from eks_ml_pipeline import AutoencoderModelDish5g
from eks_ml_pipeline import PcaModelDish5g


# functions to mark slow tests and skip them.
def pytest_addoption(parser):
    """
    Parse pytest to read --slow in testing
    """
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="run (slow) performance tests",
    )


def pytest_configure(config):
    """
    Configure markers that may be needed in our testing framework
    """
    config.addinivalue_line(
        "markers", "slow: mark test as a (potentially slow) performance test"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify items from config
    """
    if config.getoption("--slow"):
        return
    skip_perf = pytest.mark.skip(reason="need --slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_perf)


@pytest.fixture(scope="module")
def bucket_name():
    """
    Get bucket name from the github workflow runner secrets
    """
    BUCKET_NAME = os.environ.get("BUCKET_NAME_PYTEST") 
    return BUCKET_NAME


@pytest.fixture(scope="module")
def ae_train_input(bucket_name):
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

    # ********************************************#
    # ********** initialize model class **********#
    # ********************************************#

    features_df = get_features(feature_group_name, feature_input_version)
    model_parameters = features_df["model_parameters"].iloc[0]

    # Initialize autoencoder model class with specific parameters
    encode_decode_model = AutoencoderModelDish5g(
        time_steps=model_parameters["time_steps"],
        batch_size=model_parameters["batch_size"],
        epochs=1,
    )

    # File flags
    upload_zip = True
    upload_onnx = True
    upload_npy = False
    clean_local_folder = True

    return [
        encode_decode_model,
        [feature_group_name, feature_input_version],  # feature_selection,
        [data_bucketname, train_data_filename, test_data_filename],  # data_locations
        [
            save_model_local_path,
            model_bucketname,
            model_filename,
        ],  # save_model_locations,
        [upload_zip, upload_onnx, upload_npy, clean_local_folder],  # file_flags
    ]


@pytest.fixture(scope="module")
def ae_test_input(bucket_name):
    """
    Create inputs to evaluate the desired model.
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
    model_name = "test_autoencoder_ad"
    model_version = "v0.0.0"

    # Define model filename and path
    model_filename = "_".join(
        [
            model_name,
            "model",
            model_version,
            train_data_filename.split(".")[-2],  # all preceeding extension
        ]
    )

    # ********************************************#
    # ********** initialize model class **********#
    # ********************************************#

    features_df = get_features(feature_group_name, feature_input_version)
    model_parameters = features_df["model_parameters"].iloc[0]

    # Initialize autoencoder model class with specific parameters
    encode_decode_model = AutoencoderModelDish5g(
        time_steps=model_parameters["time_steps"],
        batch_size=model_parameters["batch_size"],
        epochs=1,
    )

    # File flags
    upload_zip = True
    upload_onnx = True
    upload_npy = False
    clean_local_folder = True

    return [
        encode_decode_model,
        [feature_group_name, feature_input_version],  # feature_selection,
        [data_bucketname, train_data_filename, test_data_filename],  # data_locations
        [
            save_model_local_path,
            model_bucketname,
            model_filename,
        ],  # save_model_locations,
        [upload_zip, upload_onnx, upload_npy, clean_local_folder],  # file_flags
    ]


@pytest.fixture(scope="module")
def pca_train_input(bucket_name):
    """
    Create inputs to train the desired model.
    Includes all of bucket versioning and model versioning needed
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

    # *****************************************************#
    # ********** data and model input parameters **********#
    # *****************************************************#

    ##generate pipeline input params for pca

    # feature_selection
    feature_group_name = "pytest_pca_ad"
    feature_input_version = "v0.0.1"

    # data_locations
    data_bucketname = bucket_name
    train_data_filename = "pcaDummyDataTrain.npy"
    test_data_filename = "pcaDummyDataTest.npy"

    # save_model_locations
    save_model_local_path = "../test_pca.npy"
    model_bucketname = bucket_name
    model_name = "train_pca_ad"
    model_version = "v0.0.0Pytest"
    # Define model filename
    model_filename = "_".join(
        [
            model_name,
            "model",
            model_version,
            train_data_filename.split(".")[-2],  # All preceeding extension.
        ]
    )

    # ********************************************#
    # ********** initialize model class **********#
    # ********************************************#

    features_df = get_features(feature_group_name, feature_input_version)
    features = features_df["feature_name"].to_list()
    model_parameters = features_df["model_parameters"].iloc[0]

    # Initialize pca model
    encode_decode_model = PcaModelDish5g(
        num_of_features=len(features),
        timesteps_per_slice=model_parameters["time_steps"],
    )

    # File flags
    upload_zip = False
    upload_onnx = False
    upload_npy = True
    clean_local_folder = True

    return [
        encode_decode_model,
        [feature_group_name, feature_input_version],  # feature_selection,
        [data_bucketname, train_data_filename, test_data_filename],  # data_locations
        [
            save_model_local_path,
            model_bucketname,
            model_filename,
        ],  # save_model_locations,
        [upload_zip, upload_onnx, upload_npy, clean_local_folder],  # file_flags
    ]


@pytest.fixture(scope="module")
def pca_test_input(bucket_name):
    """
    Create inputs to evaluate the desired model.
    Includes all of bucket versioning and model versioning needed
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

    # *****************************************************#
    # ********** data and model input parameters **********#
    # *****************************************************#

    ##generate pipeline input params for pca

    # feature_selection
    feature_group_name = "pytest_pca_ad"
    feature_input_version = "v0.0.1"

    # data_locations
    data_bucketname = bucket_name
    train_data_filename = "pcaDummyDataTrain.npy"
    test_data_filename = "pcaDummyDataTest.npy"

    # save_model_locations
    save_model_local_path = "../test_pca.npy"
    model_bucketname = bucket_name
    model_name = "test_pca_ad"
    model_version = "v0.0.0"
    # Define model filename
    model_filename = "_".join(
        [
            model_name,
            "model",
            model_version,
            train_data_filename.split(".")[-2],  # All preceeding extension.
        ]
    )

    # ********************************************#
    # ********** initialize model class **********#
    # ********************************************#

    features_df = get_features(feature_group_name, feature_input_version)
    features = features_df["feature_name"].to_list()
    model_parameters = features_df["model_parameters"].iloc[0]

    # Initialize pca model
    encode_decode_model = PcaModelDish5g(
        num_of_features=len(features),
        timesteps_per_slice=model_parameters["time_steps"],
    )

    # File flags
    upload_zip = False
    upload_onnx = False
    upload_npy = True
    clean_local_folder = True

    return [
        encode_decode_model,
        [feature_group_name, feature_input_version],  # feature_selection,
        [data_bucketname, train_data_filename, test_data_filename],  # data_locations
        [
            save_model_local_path,
            model_bucketname,
            model_filename,
        ],  # save_model_locations,
        [upload_zip, upload_onnx, upload_npy, clean_local_folder],  # file_flags
   ]

@pytest.fixture(scope="module")
def ae_fe_input(bucket_name):
    """
    Create inputs to pre-processing and feature engineering steps.
    Includes all of bucket versioning and model versioning needed
    as well as the file locations for a pipeline.
    Parameters
    ----------
    None

    Returns
    -------
    feature engineering inputs
        List of parameters for node rec type
        required by autoencoder feature engineering pipeline
    """

    ##feature parameters
    feature_group_name = "pytest_autoencoder_ad"
    feature_version = "v0.0.1"

    ##eks s3 bucket parameters
    partition_year = "2022"
    partition_month = "9"
    partition_day = "29"
    partition_hour = "1"
    spark_config_setup = "16gb"

    ##s3 bucket parameters
    bucket = bucket_name
    bucket_name_raw_data = bucket_name
    folder_name_raw_data = 'pytest_eks_sample_data'

    return [feature_group_name, feature_version,
            partition_year, partition_month, partition_day,
            partition_hour, spark_config_setup,
            bucket, bucket_name_raw_data, folder_name_raw_data]

# fixtures for pyspark context and session
@pytest.fixture(scope="module")
def Spark():
    
    obj = Spark_Utils()
    spark = obj.get_spark()
    return spark

@pytest.fixture(scope="module")
def Spark_context():
    
    obj = Spark_Utils()
    spark_context = obj.get_spark_context()
    return spark_context

@pytest.fixture(scope="module")
def Stop_spark():

    obj = Spark_Utils()
    obj.stop_spark_context()