"""
Streamlied train and save of Models for Encode Decode Residual Anomaly Scoring.

Contributed by Evgeniya Dontsova, Vinayak Sharma, and David Cherney
MSS Dish 5g - Pattern Detection
"""

import numpy as np
import tf2onnx
from .utilities import S3Utilities

class ModelTraining:
    """A class for training models and storing results.

    Parameters
    ------
    training_inputs : List
        Intended to be passed from the functions defined in the module
        named training_input.py.

    outputs
    -------
    None
    """
    def __init__(self, training_inputs):
        # model and feature specification
        self.encode_decode_model = training_inputs[0]
        self.feature_group_name = training_inputs[1]
        self.feature_input_version = training_inputs[2]
        # data locations
        self.data_bucketname = training_inputs[3]
        self.train_data_filename = training_inputs[4]
        self.test_data_filename = training_inputs[5]
        # save model locations
        self.save_model_local_path = training_inputs[6]
        self.model_bucketname = training_inputs[7]
        self.model_filename = training_inputs[8]
        # other
        self.model = None
        self.s3_utilities = None
        print(f'You are all set up to train a\n {self.encode_decode_model}'
              +'\nusing the .train method.')


    def initialize_s3(self):
        """Initialize s3 utilities class"""
        self.s3_utilities = S3Utilities(bucket_name = self.data_bucketname,
                               model_name = self.feature_group_name,
                               version = self.feature_input_version)

    def load_train_data(self):
        """Load training data: read from s3 bucket"""
        self.initialize_s3()
        training_tensor = self.s3_utilities.read_tensor(folder = "data",
            type_ = "tensors", file_name = self.train_data_filename)
        return training_tensor

    def ensure_np_type(self):
        """Additional data cleaning: converting everything into np.float32"""
        training_tensor = self.load_train_data()
        training_tensor = np.asarray(training_tensor).astype(np.float32)
        return training_tensor

    def train(self):
        """Train model on the object's training dataset.

        Parameters
        ----------
            None

        Returns
        -------
            None
        """
        training_tensor = self.ensure_np_type()
        model = self.encode_decode_model.fit(training_tensor)
        print('\nModel is trained.')
        ###Save model
        self.model = model
        print('\nModel is saved in memory as the attribute .model.')
        self.encode_decode_model.save_model(self.save_model_local_path)
        print(f'\nModel is saved locally in {self.save_model_local_path}.')
        print('\nTo save in S3 use the .save_to_s3 method.')


    def save_to_s3(self,
            upload_zip=False, upload_onnx=False, upload_npy=False,
            delete_local = True):
        """Save a trained model object to s3 bucket.

        Parameters
        ----------
        upload_zip : Bool (optional, default=False)
            If True than upload a .zip file to S3.

        upload_onnx : Bool (optional, default=False)
            If True than upload a .onyx file to S3.

        upload_npy : Bool (optional, default=False)
            If True than upload a .npy file to S3.

        delete_local : Bool (optional, default=True)
            If True than delete the locally saved model.

        Returns
        -------
        None
        """

        # Check that model exists and thus can be saved.
        if self.model is None:
            print('A model must be trained before it is saved.')
            return

        if upload_zip:
            #save zipped model object to s3 bucket
            self.s3_utilities.zip_and_upload(local_path = self.save_model_local_path,
                                    folder = "models",
                                    type_ = "zipped_models",
                                    file_name = self.model_filename + ".zip")
        if upload_onnx:
            # Save onnx model object to s3 bucket.
            save_model_local_path_onnx = (self.save_model_local_path + '/'
                                          + self.model_filename + ".onnx")
            # Save model locally in .onnx format.
            tf2onnx.convert.from_keras(
                self.encode_decode_model.nn,
                output_path = save_model_local_path_onnx)
            self.s3_utilities.upload_file(
                local_path = save_model_local_path_onnx,
                bucket_name = self.model_bucketname,
                key = '/'.join([self.feature_group_name,
                                 self.feature_input_version,
                                 "models", "onnx_models",
                                 self.model_filename + ".onnx"]
                               )
                                )

        if upload_npy:
            # Save npy model object to s3 bucket.
            self.s3_utilities.write_tensor(tensor = self.model,
                                  folder = "models",
                                  type_ = "npy_models",
                                  file_name = self.model_filename + ".npy")
            print('\nModel uploaded to S3 as .npy type.')

        if delete_local:
            # Delete locally saved model
            self.encode_decode_model.clean_model(self.save_model_local_path)
