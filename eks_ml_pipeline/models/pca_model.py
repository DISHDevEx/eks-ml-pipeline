import os
import logging
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

"""
Contributed by Vinayak Sharma and David Cherney
MSS Dish 5g - Pattern Detection
Groundwork to help us self heal

this models serves to provide MSS with a tool to excavate anomalies in an unsupervised fashion

"""

class PcaModelDish5g():
    """
    @:constructor num_of_features, number_of_temporal_slices, timesteps_per_slice, n_modes_to_delete

        num_of_features is the number of features per sample
        number of temporal slices is a hyperparameter that comes from the Two Time Theory
        timesteps_per_slice is time steps per a temporal slice
        train_valid_ratio indicates the training and validation split crafted from the training set. IE the input dataset will be split for training and validation

    @:returns object of class
    """


    def __init__(self,num_of_features =3, number_of_temporal_slices = 1, timesteps_per_slice = 25, n_modes_to_delete=1):

        ## super().__init__() allows for inheritence amongst child classes
        super().__init__()

        ##define num_of_features
        self.num_of_features = num_of_features

        ##define number of temporal slices
        self.number_of_temporal_slices = number_of_temporal_slices

        ##set timesteps
        self.timesteps_per_slice = timesteps_per_slice

        ##create a results df
        self.results = None

        ##create an x_train
        self.x_train: np.ndarray = None

        ##create an x_train
        self.n_modes_to_delete= n_modes_to_delete

        ## CREATE the vectors for saving
        self.vs = None

        self.ss = StandardScaler()

    def load_model(self,filename = 'vs.npy'):
        """
        @:param vs: file path with model weights
        Takes model weights and:
            - loads them
        @:returns nothing
        """
        self.vs = np.load(filename)

    def save_model(self, filename = 'vs.npy'):
        """
        @:param filename
        Takes model weights and:
            - saves them in a .npy file
        @:returns nothing
        """
        np.save(filename, self.vs)


    def clean_model(self, filename):
        """
        @:param filename: name of file for locally saved model
        @:returns nothing
        """
        os.remove(filename)
        print(f"\n***Locally saved model in {filename} was succesfully deleted.***\n")

    def two_time_slice(self, samples):
        """
        @:param samples: training data or data set with shape [samples,ts,features]
        Takes training set and:
            - reshapes it with two time approach
        @:returns r3 and r4 where r4 is a slicing of the input with shape [samples, H,T,F]  and r3 is a concatenation of r4 over axis 1 to stack the slices, and so is of shape [samples*H , T, F]
        """
        rank4_sliced = samples.reshape(samples.shape[0],
                                       self.number_of_temporal_slices,
                                       self.timesteps_per_slice,
                                       self.num_of_features,
                                      )[:]

        rank3_sliced = rank4_sliced.reshape(samples.shape[0]*self.number_of_temporal_slices,
                                            self.timesteps_per_slice,
                                            self.num_of_features,
                                           )[:]
        return rank3_sliced, rank4_sliced

    def fit(self, x_train):
        """
        @:param x_train: training data
        Takes training set and:
            - fits the model
        @:returns residuals,ed_errors,vs
        """
        ##log that the pca model training has begun##
        logging.info("PCA model training started")

        ##conditional temporal slicing check##
        if( self.number_of_temporal_slices >1 ):
            trainX_slices_as_samples, trainX_sliced  =  self.two_time_slice(x_train)
        else: trainX_slices_as_samples = x_train

        ### initializing, just for shape###
        trainX_slices_as_samples_ss = trainX_slices_as_samples[:]
        trainX_slices_as_samples_ss_encoded = np.zeros(shape = (self.timesteps_per_slice - self.n_modes_to_delete ,trainX_slices_as_samples.shape[0],self.num_of_features ))
        trainX_slices_as_samples_ss_decoded = np.zeros(shape = (trainX_slices_as_samples.shape[0],self.timesteps_per_slice,self.num_of_features))


        ### apply standard scalar one feature at a time:###
        for i in range(self.num_of_features):
            trainX_slices_as_samples_ss[:,:,i] = self.ss.fit_transform(trainX_slices_as_samples[:,:,i])

        ### Apply PCA and get the principle vectors for each feature ###
        vs = [] # vs is a list, one component for each feature, with Principal Vectors in a matrix V
        pca = PCA(n_components = self.timesteps_per_slice)
        for i in range(self.num_of_features):
            pca.fit(trainX_slices_as_samples_ss[:,:,i]) # fits vectors across (n_samples, n_features)
            vs.append(pca.components_) # orthonormal basis of sample space in order of variance explained

        self.vs = vs

        ##encode and decode using pca##
        for i in range(self.num_of_features):
            trainX_slices_as_samples_ss_encoded[:,:,i] = np.matmul(vs[i][:-self.n_modes_to_delete,:],
                                                          trainX_slices_as_samples_ss[:,:,i].T)
            trainX_slices_as_samples_ss_decoded[:,:,i] = np.matmul(vs[i][:-self.n_modes_to_delete,:].T,
                                                          trainX_slices_as_samples_ss_encoded[:,:,i]).T

        ##calculate residuals and errors##
        residuals = trainX_slices_as_samples_ss - trainX_slices_as_samples_ss_decoded
        ed_errors = np.mean(np.absolute(residuals),
                               axis =1, # across timestamps
                           )
        #return residuals, errors, and vs
        #return residuals, ed_errors, self.vs
        return self.vs

    def predict(self, x_test):
        """
        @:param x_test: test data. Must be of shape [ samples, timesteps, features]

        -apply inferencing

        @:returns reconstructions,residuals
        """
        ##log that the pca model training has begun##
        logging.info("PCA model testing started")

        ##two time slicing conditional check##
        if( self.number_of_temporal_slices >1 ):
            testX_slices_as_samples, testX_sliced  =  self.two_time_slice(x_train)
        else: testX_slices_as_samples = x_test

        ### init for shape ##
        testX_slices_as_samples_ss = testX_slices_as_samples[:]
        testX_slices_as_samples_ss_encoded = np.zeros(shape = (self.timesteps_per_slice \
                                                               - self.n_modes_to_delete ,
                                                               testX_slices_as_samples.shape[0],self.num_of_features ))
        testX_slices_as_samples_ss_decoded = np.zeros(shape = (testX_slices_as_samples.shape[0],
                                                               self.timesteps_per_slice,self.num_of_features))

        ##apply standard scaler###
        for i in range(self.num_of_features):
            testX_slices_as_samples_ss[:,:,i] = self.ss.fit_transform(testX_slices_as_samples[:,:,i])

        ##encode and decode using pca##
        for i in range(self.num_of_features):
            testX_slices_as_samples_ss_encoded[:,:,i] = np.matmul(self.vs[i][:-self.n_modes_to_delete,:],
                                                          testX_slices_as_samples_ss[:,:,i].T)
            testX_slices_as_samples_ss_decoded[:,:,i] = np.matmul(self.vs[i][:-self.n_modes_to_delete,:].T,
                                                          testX_slices_as_samples_ss_encoded[:,:,i]).T

        residuals = testX_slices_as_samples_ss - testX_slices_as_samples_ss_decoded


        #return reconstructions and residuals
        return [testX_slices_as_samples_ss_decoded, residuals]