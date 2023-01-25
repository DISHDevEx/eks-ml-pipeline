"""
Streamlied train and save of Models for Encode Decode Residual Anomaly Scoring.

Contributed by Evgeniya Dontsova, Vinayak Sharma, and David Cherney
MSS Dish 5g - Pattern Detection
"""

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

    def load_train_data(self):
        """Load training data: read from s3 bucket"""
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

        if self.file_flags[3]: # upload_npy:
            # Save npy model object to s3 bucket.
            self.s3_utilities.write_tensor(tensor = self.encode_decode_function,
                                  folder = "models",
                                  type_ = "npy_models",
                                  file_name = self.save_model_locations[2] + ".npy")
            print('\nModel uploaded to S3 as .npy type.')

        ## Delete the locally saved model.
        if self.file_flags[0]: # clean_local_folder:
            # Delete locally saved model
            self.encode_decode_model.clean_model(self.save_model_locations[0])
            print(f'Local file {self.save_model_locations[0]} deleted.')

    #####################################################
    ######## Methods for evaluation of the model:########
    #####################################################

    def load_test_data(self):
        """Load training data: read from s3 bucket"""
        testing_tensor = self.s3_utilities.read_tensor(
            folder = "data",
            type_ = "tensors",
            file_name = self.data_locations[2] # test_data_filename
            )
        # ensure np type
        testing_tensor = np.asarray(testing_tensor).astype(np.float32)
        return testing_tensor

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
        testing_tensor = self.load_test_data()

#         # if model is not in memory and all booleans are false, print a notice
#         if self.encode_decode_function is None:
#             print("No Encode-Decode function is in memory for evaluation.")
#             return

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
            
    #######
    ### methods to upload trained models

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

        # Local file is no longer needed; delete it.
        if self.file_flags[3]: # clean_local_folder:
            self.encode_decode_model.clean_model(
                self.save_model_locations[0] #save_model_local_path
                )
