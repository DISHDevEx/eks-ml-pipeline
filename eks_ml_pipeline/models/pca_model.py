import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from tensorflow import keras
# import tensorflow as tf
# from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from numpy import save
from numpy import load

"""
Contributed by Vinayak Sharma and David Cherney
MSS Dish 5g - Pattern Detection
Groundwork to help us self heal

this models serves to provide MSS with a tool to excavate anomalies in an unsupervised fashion

"""

class pca_model_dish_5g():
    """
    @:constructor num_of_features, number_of_temporal_slices, timesteps_per_slice, n_modes_to_delete
    
    @:returns object of class 
    """
    ## num_of_features, number_of_temporal_slices, timesteps_per_slice, n_modes_to_delete
    ## num_of_features is the number of features per sample
    ## number of temporal slices is a hyperparameter that comes from the Two Time Theory
    ## timesteps_per_slice is 
    ## train_valid_ratio indicates the training and validation split crafted from the training set. IE the input dataset will be split for training and validation
    
    def __init__(self,num_of_features =3, number_of_temporal_slices = 1, timesteps_per_slice = 25, n_modes_to_delete=1):
        
        ## super().__init__() allows for inheritence amongst child classes
        super().__init__()
        
        ##define num_of_features
        self.num_of_features = num_of_features
            
        ##define number of temporal slices
        self.number_of_temporal_slices = number_of_temporal_slices
            
        ##set timesteps
        self.timesteps_per_slice = timesteps_per_slice
        
        self.N = number_of_temporal_slices * timesteps_per_slice
        
        ##create a results df
        self.results = None
        
        ##create an x_train
        self.x_train: np.ndarray = None
      
        ##create an x_train
        self.n_modes_to_delete= n_modes_to_delete
            
        ## CREATE the vectors for saving
        self.vs = None
        
        ##create encode and decode for 
        self.encode_decode_maps = None
        
        self.ss = StandardScaler()
        
    def load_in_vs(self,vs):
        """
        @:param vs: file path with model weights
        Takes model weights and:
            - loads them
        @:returns nothing
        """
        self.vs = load(vs)
        
    def save_vs(self, filename = 'vs.npy'):
        """
        @:param filename
        Takes model weights and:
            - saves them in a .npy file
        @:returns nothing
        """
        save(filename, self.vs)
        
    def two_time_slice(self, samples):
        """
        @:param samples: training data or data set with shape [samples,ts,features]
        Takes training set and:
            - reshapes it with two time approach
        @:returns a rank3 sliced and rank4 sliced tensor 
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
        @:returns residuals,ed_errors,self.encode_decode_maps
        """
        ##log that the autoencoder model training has begun
        logging.info("Autoencoder model training started")
        
        trainX_slices_as_samples, trainX_sliced  =  self.two_time_slice(x_train)
        
        # initializing, just for shape. The [:] is needed to have a copy instead of a view
        trainX_slices_as_samples_ss = trainX_slices_as_samples[:] 
        # initialize, since I have to do matmult by component
        trainX_slices_as_samples_ss_encoded = np.zeros(shape = (self.timesteps_per_slice - self.n_modes_to_delete ,trainX_slices_as_samples.shape[0],self.num_of_features ))
        trainX_slices_as_samples_ss_decoded = np.zeros(shape = (trainX_slices_as_samples.shape[0],self.timesteps_per_slice,self.num_of_features))
        
        # one feature at a time:
        for i in range(self.num_of_features):
            trainX_slices_as_samples_ss[:,:,i] = self.ss.fit_transform(trainX_slices_as_samples[:,:,i]) 
            
        # a list, one component for each feature, with Principal Vectors in a matrix V 
        vs = []
        pca = PCA(n_components = self.timesteps_per_slice)
        for i in range(self.num_of_features):
            pca.fit(trainX_slices_as_samples_ss[:,:,i]) # fits vectors across (n_samples, n_features)
            vs.append(pca.components_) # orthonormal basis of sample space in order of variance explained
         
        self.vs = vs
        
        ##create the encode decode maps
        self.encode_decode_maps = [np.matmul( vs[i][:-self.n_modes_to_delete,:].T, vs[i][:-self.n_modes_to_delete,:] ) for i in range(self.num_of_features) ]
 
        ##encode and decode using pca
        for i in range(self.num_of_features):
            trainX_slices_as_samples_ss_encoded[:,:,i] = np.matmul(vs[i][:-self.n_modes_to_delete,:], 
                                                          trainX_slices_as_samples_ss[:,:,i].T)
            trainX_slices_as_samples_ss_decoded[:,:,i] = np.matmul(vs[i][:-self.n_modes_to_delete,:].T, 
                                                          trainX_slices_as_samples_ss_encoded[:,:,i]).T

        # calculate residuals and errors
        residuals = trainX_slices_as_samples_ss - trainX_slices_as_samples_ss_decoded
        residuals_reshaped = residuals.reshape(-1,self.N,self.num_of_features)
        ed_errors = np.linalg.norm(residuals,
                    ord =1, # MAE, as in the rules
                    axis=1)
        
        return residuals_reshaped,ed_errors,self.encode_decode_maps

    def test(self, x_test):
        """
        @:param x_test: test data. Must be of shape [ samples, timesteps, features]
        
        -apply inferencing
        
        @:returns decoding,residuals
        """
        testX_slices_as_samples, testX_sliced = self.two_time_slice(x_test)
        
        testX_slices_as_samples_ss = testX_slices_as_samples[:]
        
                # initialize, since we have to do matmul by feature
        testX_slices_as_samples_ss_encoded = np.zeros(shape = (self.timesteps_per_slice - self.n_modes_to_delete ,testX_slices_as_samples.shape[0],self.num_of_features ))
        testX_slices_as_samples_ss_decoded = np.zeros(shape = (testX_slices_as_samples.shape[0],self.timesteps_per_slice,self.num_of_features))
        
        ##apply standard scaler
        for i in range(self.num_of_features):
            testX_slices_as_samples_ss[:,:,i] = self.ss.fit_transform(testX_slices_as_samples[:,:,i])
    
        ##encode and decode using pca
        for i in range(self.num_of_features):
            testX_slices_as_samples_ss_encoded[:,:,i] = np.matmul(self.vs[i][:-self.n_modes_to_delete,:], 
                                                          testX_slices_as_samples_ss[:,:,i].T)
            testX_slices_as_samples_ss_decoded[:,:,i] = np.matmul(self.vs[i][:-self.n_modes_to_delete,:].T, 
                                                          testX_slices_as_samples_ss_encoded[:,:,i]).T
            
        residuals = testX_slices_as_samples_ss - testX_slices_as_samples_ss_decoded
        
        residuals_reshaped = residuals.reshape(-1,self.N,self.num_of_features)
        
        testX_slices_as_samples_ss_decoded_reshaped = testX_slices_as_samples_ss_decoded.reshape(-1,self.N,self.num_of_features)
        
        return testX_slices_as_samples_ss_decoded_reshaped,residuals_reshaped