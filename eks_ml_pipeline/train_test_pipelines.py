"""
Streamlied train and save of Models for Encode Decode Residual Anomaly Scoring.
"""

import os
import numpy as np
import tf2onnx
from .utilities import S3Utilities

class TrainTestPipelines:
    """A class for training and testing the models in the directory modules,
    and storing results.

    Parameters
    ------
    training_inputs : List
        Intended to be one of the functions in inputs/training_input.py.

    outputs
    -------
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

        # file_flags = [upload_zip, upload_onnx, upload_npy, clean_local_folder]
        self.file_flags = training_inputs[4]

        # other
        self.encode_decode_function = None
        self.s3_utilities = None
        self.initialize_s3()
        print(f'You are all set up to train a\n {self.encode_decode_model}'
              +'\nusing the .train method.')


    def initialize_s3(self):
        """Initialize s3 utilities class"""
        self.s3_utilities = S3Utilities(
            bucket_name = self.data_locations[0], # data_bucketname
            model_name = self.feature_selection[0], # feature_group_name
            version = self.feature_selection[1], # feature_input_version
            )

    def train(self):
        """Train model on the object's training dataset.

        Parameters
        ----------
            None

        Returns
        -------
            None
        """
        training_tensor = self.load_data(data_purpose = 'training')
        # for PCA it is the output of .fit that is saved to S3.
        encode_decode_function = self.encode_decode_model.fit(training_tensor)
        print('\nModel is trained.')
        ###Save model
        self.encode_decode_function = encode_decode_function
        self.encode_decode_model.save_model(
            self.save_model_locations[0] # save_model_local_path
            )
        self.save_to_s3()


    def save_to_s3(self):
        """Save a trained model object to s3 bucket.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if self.file_flags[0]: # upload_zip:
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
        if self.file_flags[1]: # upload_onnx:
            # Save onnx model object to s3 bucket.
            save_model_local_path_onnx = (self.save_model_locations[0] + '/'
                                          + self.save_model_locations[2]
                                          + ".onnx")
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
                     ) # end of .join args
                ) # end of upload_file args

        if self.file_flags[2]: # upload_npy:
            # Save npy model object to s3 bucket.
            self.s3_utilities.write_tensor(
                tensor = self.encode_decode_function,
                folder = "models",
                type_ = "npy_models",
                file_name = (self.save_model_locations[2] + ".npy")
                )
            print('\nModel uploaded to S3 as .npy type.')

        ## Delete the locally saved model.
        if self.file_flags[3]: # clean_local_folder:
            # Delete locally saved model
            self.encode_decode_model.clean_model(self.save_model_locations[0])
            print(f'Local file {self.save_model_locations[0]} deleted.')

    #####################################################
    ######## Methods for evaluation of the model:########
    #####################################################

    def test(self):
        """Evaluate model on test data.

        The test data was specified in creation of the class object.

        Parmeters
        ---------
        clean_local_folder: bool
            Flag to delete or keep locally saved model directory or files.

        Outputs
        -------
            None
        """
        # Load the model
        # to make the methods of self.encode_decode_model available.
        self.load_model()

        ###Load training data: read from s3 bucket
        testing_tensor = self.load_data(data_purpose = 'testing')


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

        #Delete locally saved model
        if self.file_flags[3]: # clean_local_folder

            self.encode_decode_model.clean_model(
                self.save_model_locations[0] #save_model_local_path
                )
            if self.file_flags[0]: # upload_zip

                path = (self.save_model_locations[0] #save_model_local_path
                        + '.zip')
                os.remove(path)
                print(f"\n***Locally saved {path} was succesfully deleted.***\n")

    #####################################################
    ######Methods to Upload Trained Models and Data######
    #####################################################
    ###

    def load_model(self):
        """
        Read a trained model in from S3.

        Parameters
        ----------
        upload_zip : bool
            upload a trained model from .zip format (for autoencoders)

        upload_npy : bool
            upload a model from .npy format (for PCA models)

        """
        ### Load trained model: read from s3 bucket

        # save file locally so that if it is a zip it can be unzipped.
        if self.file_flags[0]:# upload_zip:
            self.s3_utilities.download_zip(
                local_path = (self.save_model_locations[0] # save_model_local_path
                              + '.zip'),
                folder = "models",
                type_ = "zipped_models",
                file_name = (self.save_model_locations[2] # model_filename
                             + '.zip')
                )

            self.s3_utilities.unzip(
                path_to_zip = (self.save_model_locations[0] # save_model_local_path
                               + '.zip'),
                extract_location = self.save_model_locations[0] # save_model_local_path
                )


        if self.file_flags[2]: # upload_npy:
            load_tensor = self.s3_utilities.read_tensor(
                folder = "models",
                type_ = "npy_models",
                file_name = (self.save_model_locations[2] # model_filename
                             + ".npy")
                )

            np.save(self.save_model_locations[0], # save_model_local_path,
                    load_tensor
                    )

        ## While model file is saved in local path,
        ## use model object's load_model method.
        ## This makes the methods of self.encode_decode_model available.
        self.encode_decode_model.load_model(
            self.save_model_locations[0], # save_model_local_path
            )


    def load_data(self, data_purpose = 'training'):
        """
        Load test or training data from s3 bucket.

        Parameters
        ----------
        data_purpose : string
            'training' or 'testing'
        Returns
        -------
        tensor : np.array
            A tensor of data built for the designated purpose.
        """
        # Determine if training or testing data is requested.
        if data_purpose == 'training':
            f_name = self.data_locations[1] # train_data_filename
        elif data_purpose == 'testing':
            f_name = self.data_locations[2] # test_data_filename

        tensor = self.s3_utilities.read_tensor(
            folder = "data",
            type_ = "tensors",
            file_name = f_name
            )
        # ensure np type
        tensor = np.asarray(tensor).astype(np.float32)
        return tensor
