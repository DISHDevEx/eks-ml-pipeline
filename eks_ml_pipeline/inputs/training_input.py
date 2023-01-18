"""
A script to feed inputs to the training pipeline.

To use
1. choose the function below corresponding to the model of your choice
2. modify the data and model imput paramters as needed
3. feed the input function to the class in training.py

e.g. use

ModelTraining(node_pca_input())

Contributed by Evgeniya Dontsova, Vinayak Sharma, and David Cherney
MSS Dish 5g - Pattern Detection
"""

from msspackages import get_features
from ..models import AutoencoderModelDish5g
from ..models import PcaModelDish5g

# from ..secrets import data_bucketname as data_bucket
# from ..secrets import model_bucketname as model_bucket
# from ..secrets import secrets


def node_autoencoder_input():
    """
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
    feature_group_name = "node_autoencoder_ad"
    feature_input_version = "v0.0.2"

    # data_locations
    data_bucketname = 'dish-5g.core.pd.g.dp.eks.logs.e'
    train_data_filename = "training_2022_9_29_1.npy"
    test_data_filename = "testing_2022_9_29_1.npy"

    # save_model_locations
    save_model_local_path = "../node_autoencoder"
    model_bucketname = 'dish-5g.core.pd.g.dp.eks.logs.e'
    model_name = 'node_autoencoder_ad'
    model_version = "v0.0.1-test"
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
        time_steps=model_parameters["time_steps"],
        batch_size=model_parameters["batch_size"],
        epochs=1
        )

    return [encode_decode_model,
            [feature_group_name,feature_input_version], # feature_selection,
            [data_bucketname, train_data_filename, test_data_filename], # data_locations
            [save_model_local_path, model_bucketname, model_filename,]# save_model_locations,
            ]

def node_pca_input():
    """
    Parameters
    ----------
    None

    Returns
    -------
    training_inputs
        list of parameters for node rec type
        required by pca model
        training pipeline
    """

    #*****************************************************#
    #********** data and model input parameters **********#
    #*****************************************************#

    ##generate pipeline input params for pca

    # feature_selection
    feature_group_name = "node_pca_ad"
    feature_input_version = "v0.0.1"

    # data_locations
    data_bucketname = 'dish-5g.core.pd.g.dp.eks.logs.e'
    train_data_filename = "training_2022_9_29_1.npy"
    test_data_filename = "testing_2022_9_29_1.npy"

    # save_model_locations
    save_model_local_path = "../node_pca.npy"
    model_bucketname = 'dish-5g.core.pd.g.dp.eks.logs.e'
    model_name = "node_pca_ad"
    model_version = "v0.0.1-test"

    #Define model filename
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
    features = features_df["feature_name"].to_list()
    model_parameters = features_df["model_parameters"].iloc[0]

    #Initialize pca model
    encode_decode_model = PcaModelDish5g(
        num_of_features = len(features),
        timesteps_per_slice = model_parameters["time_steps"]
        )

    return [encode_decode_model,
            [feature_group_name,feature_input_version], # feature_selection,
            [data_bucketname, train_data_filename, test_data_filename], # data_locations
            [save_model_local_path, model_bucketname, model_filename,]# save_model_locations,
            ]

def pod_autoencoder_input():
    """
    Parameters
    ----------
    None

    Returns
    -------
    training_inputs
        list of parameters for pod rec type
        required by autoencoder model
        training pipeline
    """

    #*****************************************************#
    #********** data and model input parameters **********#
    #*****************************************************#

    # feature_selection
    feature_group_name = "pod_autoencoder_ad"
    feature_input_version = "v0.0.2"

    # data+locations
    data_bucketname = 'dish-5g.core.pd.g.dp.eks.logs.e'
    train_data_filename = "training_2022_9_9_1.npy"
    test_data_filename = "testing_2022_9_9_1.npy"

    # save_model_locations
    save_model_local_path = "../pod_autoencoder"
    model_bucketname = 'dish-5g.core.pd.g.dp.eks.logs.e'
    model_name = 'pod_autoencoder_ad'
    model_version = "v0.0.1-test"
    #Define model filename
    model_filename = '_'.join(
        [model_name,
         "model",
         model_version,
         train_data_filename.split(".")[-2] # All preceeding extension.
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

    return [encode_decode_model,
            [feature_group_name,feature_input_version], # feature_selection,
            [data_bucketname, train_data_filename, test_data_filename], # data_locations
            [save_model_local_path, model_bucketname, model_filename,]# save_model_locations,
            ]

def pod_pca_input():
    """
    Parameters
    ----------
    None

    Returns
    -------
    training_inputs
        list of parameters for pod rec type
        required by pca model
        training pipeline
    """

    #*****************************************************#
    #********** data and model input parameters **********#
    #*****************************************************#

    ##generate pipeline input params for pca

    # feature_selection
    feature_group_name = "pod_pca_ad"
    feature_input_version = "v0.0.1"

    # data_locations
    data_bucketname = 'dish-5g.core.pd.g.dp.eks.logs.e'
    train_data_filename = "training_2022_9_9_1.npy"
    test_data_filename = "testing_2022_9_9_1.npy"

    # save_model_locations
    save_model_local_path = "../pod_pca.npy"
    model_bucketname = 'dish-5g.core.pd.g.dp.eks.logs.e'
    model_name = "pod_pca_ad"
    model_version = "v0.0.1-test"
    #Define model filename
    model_filename = '_'.join(
        [model_name,
         "model",
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

    return [encode_decode_model,
            [feature_group_name,feature_input_version], # feature_selection,
            [data_bucketname, train_data_filename, test_data_filename], # data_locations
            [save_model_local_path, model_bucketname, model_filename,]# save_model_locations,
            ]

def container_autoencoder_input():
    """
    Parameters
    ----------
    None

    Returns
    -------
    training_inputs
        list of parameters for container rec type
        required by autoencoder model
        training pipeline
    """

    #*****************************************************#
    #********** data and model input parameters **********#
    #*****************************************************#

    # feature_selection
    feature_group_name = "container_autoencoder_ad"
    feature_input_version = "v0.0.1"

    # data_locations
    data_bucketname = 'dish-5g.core.pd.g.dp.eks.logs.e'
    train_data_filename = "training_2022_5_5_1.npy"
    test_data_filename = "testing_2022_5_5_1.npy"

    # save_model_locations
    save_model_local_path = "../container_autoencoder"
    model_bucketname = 'dish-5g.core.pd.g.dp.eks.logs.e'
    model_name = 'container_autoencoder_ad'
    model_version = "v0.0.1-test"
    #Define model filename
    model_filename = '_'.join(
        [model_name,
         "model",
         model_version,
         train_data_filename.split(".")[-2] # All preceeding extension.
         ])

    #********************************************#
    #********** initialize model class **********#
    #********************************************#

    features_df = get_features(feature_group_name, feature_input_version)
    model_parameters = features_df["model_parameters"].iloc[0]

    #Initialize autoencoder model class with specific parameters
    encode_decode_model = AutoencoderModelDish5g(
        time_steps=model_parameters["time_steps"],
        batch_size=model_parameters["batch_size"],
        epochs=1
        )

    return [encode_decode_model,
            [feature_group_name,feature_input_version], # feature_selection,
            [data_bucketname, train_data_filename, test_data_filename], # data_locations
            [save_model_local_path, model_bucketname, model_filename,]# save_model_locations,
            ]

def container_pca_input():
    """
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
    feature_group_name = "container_pca_ad"
    feature_input_version = "v0.0.1"

    # data_locations
    data_bucketname = 'dish-5g.core.pd.g.dp.eks.logs.e'
    train_data_filename = "training_2022_5_5_1.npy"
    test_data_filename = "testing_2022_5_5_1.npy"

    # save_model_locations
    save_model_local_path = "../container_pca.npy"
    model_bucketname = 'dish-5g.core.pd.g.dp.eks.logs.e'
    model_name = "container_pca_ad"
    model_version = "v0.0.1-test"
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

    return [encode_decode_model,
            [feature_group_name,feature_input_version], # feature_selection,
            [data_bucketname, train_data_filename, test_data_filename], # data_locations
            [save_model_local_path, model_bucketname, model_filename,]# save_model_locations,
            ]
