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
        Intended to from the functions defined in the module
        named inputs/training_input.py.

    outputs
    -------
    None
    """

    def __init__(self, training_inputs):
        # model
        self.encode_decode_model = training_inputs[0]

        # feature_selection = [feature_group_name,feature_input_version]
        self.feature_selection = training_inputs[1]

        # data_locations = [data_bucketname, train_data_filename, test_data_filename]
        self.data_locations = training_inputs[2]

        # save_model_locations = [save_model_local_path, model_bucketname, model_filename,]
        self.save_model_locations = training_inputs[3]

        # other
        self.model = None
        self.s3_utilities = None
        print(f'You are all set up to train a\n {self.encode_decode_model}'
              +'\nusing the .train method.')


    def initialize_s3(self):
        """Initialize s3 utilities class"""
        self.s3_utilities = S3Utilities(
            bucket_name = self.data_locations[0], # data_bucketname
            model_name = self.feature_selection[0], # feature_group_name
            version = self.feature_selection[1], # feature_input_version
            )

    def load_train_data(self):
        """Load training data: read from s3 bucket"""
        self.initialize_s3()
        training_tensor = self.s3_utilities.read_tensor(
            folder = "data",
            type_ = "tensors",
            file_name = self.data_locations[1] # train_data_filename
            )
        # ensure np type
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
        training_tensor = self.load_train_data()
        model = self.encode_decode_model.fit(training_tensor)
        print('\nModel is trained.')
        ###Save model
        self.model = model
        print('\nModel is saved in memory as the attribute .model.')
        self.encode_decode_model.save_model(self.save_model_locations[0])
        print(f'\nModel is saved locally in {self.save_model_locations[0]}.')
        print('\nTo save in S3 use the .save_to_s3 method.'
              + '\nNote the option to delete the local copy.')


    def save_to_s3(self,
            upload_zip=False, upload_onnx=False, upload_npy=False,
            delete_local = False):
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

        # Check that a format has been chosen.
        if not (upload_zip or upload_onnx or upload_npy):
            print('Parameters specified that no files be saved to S3.')

        if upload_zip:
            try:
                #save zipped model object to s3 bucket
                self.s3_utilities.zip_and_upload(
                    local_path = self.save_model_locations[0], # save_model_local_path
                    folder = "models",
                    type_ = "zipped_models",
                    file_name = (self.save_model_locations[2] # model_filename
                                 + ".zip")
                    )
            except NotADirectoryError:
                print('The requested format, zip, is not appropriate '
                      + 'because your model is not in a folder.')
        if upload_onnx:
            # Save onnx model object to s3 bucket.
            save_model_local_path_onnx = (self.save_model_locations[0] + '/'
                                          + self.save_model_locations[2] + ".onnx")
            # Save model locally in .onnx format.
            tf2onnx.convert.from_keras(
                self.encode_decode_model.nn,
                output_path = save_model_local_path_onnx)
            self.s3_utilities.upload_file(
                local_path = save_model_local_path_onnx,
                bucket_name = self.save_model_locations[1],
                key = '/'.join(
                    [self.feature_selection[0], # feature_group_name
                     self.feature_selection[1], # feature_input_version
                     "models", "onnx_models",
                     self.save_model_locations[2] + ".onnx"]
                     )
                )

        if upload_npy:
            # Save npy model object to s3 bucket.
            self.s3_utilities.write_tensor(tensor = self.model,
                                  folder = "models",
                                  type_ = "npy_models",
                                  file_name = self.save_model_locations[2] + ".npy")
            print('\nModel uploaded to S3 as .npy type.')

        if delete_local:
            # Delete locally saved model
            self.encode_decode_model.clean_model(self.save_model_locations[0])
            print(f'Local file {self.save_model_locations[0]} deleted.')

    #####################################################
    ######## Methods for evaluation of the model:########
    #####################################################

    def load_test_data(self):
        """Load training data: read from s3 bucket"""
        self.initialize_s3() # in case user has not ititialized S3 through .train
        testing_tensor = self.s3_utilities.read_tensor(
            folder = "data",
            type_ = "tensors",
            file_name = self.data_locations[2] # test_data_filename
            )
        # ensure np type
        testing_tensor = np.asarray(testing_tensor).astype(np.float32)
        return testing_tensor

    def evaluate(self,
             clean_local_folder = False,
             ):
        """Evaluate model on test data.

        The test data was specified in creation of the class object.

        Parmeters
        ---------
        clean_local_folder: bool
            flag to delete or keep locally saved model directory or files

        Outputs
        -------
            None
        """
        ###Load training data: read from s3 bucket
        testing_tensor = self.load_test_data()

        # if model is not in memory and all booleans are false, print a notice
        if self.model is None:
            print("No mode lis in memory for evaluation.")
            return

        ###Use trained model to predict for testing tensor
        results = self.encode_decode_model.predict(testing_tensor)

        ###Save predictions
        test_data_filename = self.data_locations[2]
        for label, result in zip(['predictions', 'residuals'], results):
            print(f'Writing {test_data_filename.split(".")[-2]}_{label}.npy '
                   + 'to S3...')
            self.s3_utilities.write_tensor(
                tensor = result,
                folder = "models",
                type_ = "predictions",
                file_name = f'{test_data_filename.split(".")[-2]}_{label}.npy')

        if clean_local_folder:
            self.encode_decode_model.clean_model(
                self.save_model_locations[0] #save_model_local_path
                )
