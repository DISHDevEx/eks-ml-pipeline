import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

"""
Contributed by Vinayak Sharma and David Cherney
MSS Dish 5g - Pattern Detection
Groundwork to help us self heal

this models serves to provide MSS with a tool to excavate anomalies in an unsupervised fashion

"""

class autoencoder_model_dish_5g():
    """
    @:constructor takes in timesteps, batch size, learning rate, and a train_valid ratio
    
    @:returns object of class 
    """
    ## constructor takes in timesteps, batch size, learning rate, and a train_valid ratio
    ## timesteps is the number of time intervals inside of a sample
    ## batch size is number of  samples per iteration
    ## learning rate is the hyperparameter eta
    ## train_valid_ratio indicates the training and validation split crafted from the training set. IE the input dataset will be split for training and validation
    
    def __init__(self, time_steps=12, batch_size=6, learning_rate=0.001,
                 validation_split=.1, epochs=100, nuerons = 128, dropout_rate=.1, patience = 5):
        
        ## super().__init__() allows for inheritence amongst child classes
        super().__init__()
        
        ##init error threshold
        self.error_threshold = None
        
        ##anomaly score threshold is set to .95
        self.anomaly_score_threshold: float = 0.95
            
        ##set timesteps
        self.time_steps = time_steps
        
        ##set batch size
        self.batch_size = batch_size
        
        ##set learning rate
        self.lr = learning_rate
        
        ##create a results df
        self.results = None
        
        ##init the self nueral net to None, later it will be defined
        self.nn = None
        
        ##init the number of epochs for the nueral network to train
        self.epochs = epochs
        
        ##init the valid sploit
        self.validation_split = validation_split
        
        ##init nuerons
        self.nuerons = nuerons
        
        ##init dropout rate
        self.dropout_rate = dropout_rate
        
        ##init patience
        self.patience = patience
        
    ## this function calculates a threshold 
    @staticmethod
    def __calculate_threshold(valid_errors: np.ndarray) -> float:
        return 2 * np.max(valid_errors)
    
    ## this function inits the model
    def __initialize_nn(self, x_train):
        
        ##actually define the nueral network here. 
        ##Its a Keras sequential with an Bi-LSTM, a dropout, a repeat vector, then another Bi-LSTM, another dropout, and a timedistributed output. 
        ##The output is mapped back the number of features. 
        ##RepeatVector "Repeats the input n times."https://keras.io/api/layers/reshaping_layers/repeat_vector/
        
        ##none of the nueral network is hard coded (except for the number of nuerons b)
        self.nn  = keras.Sequential(
                    [
                        layers.Bidirectional(layers.LSTM(self.nuerons,dropout=self.dropout_rate),input_shape=(x_train.shape[1], x_train.shape[2])),
                        layers.RepeatVector(x_train.shape[1]),
                        layers.Bidirectional(layers.LSTM(self.nuerons,dropout=self.dropout_rate, return_sequences=True)),
                        layers.TimeDistributed(layers.Dense(x_train.shape[2])),
                    ]
                )
        
        ##nn compile groups layers into a model; the loss here is MAE, optimizer is Adam, learning rate is predefined
        self.nn.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr), loss="mae")
        self.nn.summary()
        
    def __calculate_pred_and_err(self, data):
        predictions = self.nn.predict(data)
        mae_loss = np.mean(np.abs(predictions - data), axis=1)
        return predictions,mae_loss
    
    def __calculate_anomaly_score(self, residuals: np.ndarray, initial_threshold: float = 1.0):
        max_resid = np.nanmax(np.array(residuals, dtype=np.float64))
        if max_resid is None:
            max_resid = 1
        scaler = MinMaxScaler(feature_range=(0, max_resid / initial_threshold))
        anom_scores = scaler.fit_transform(residuals.reshape(-1, 1))
        anom_scores[anom_scores > 1.0] = 1.0
        return anom_scores
    
    
    def save_nn(self, filename):
        """
        @:param filename: name of file to save model
            -saves nn to file
        @:returns nothing 
        """
        tf.keras.models.save_model(self.nn, filename)
    
    def load_nn(self, filename):
        """
        @:param filename: name of file to save model
        @:returns nothing 
        """
    
        self.nn = tf.keras.models.load_model(filename)
        #self.error_threshold = error_threshold
        self.trained = True

    def fit(self, x_train):
        """
        @:param x_train: training data 
        Takes training set and:
            - fits the model
        @:returns nothing 
        """
        
        ##log that the autoencoder model training has begun
        logging.info("Autoencoder model training started")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.__initialize_nn(x_train)
            history = self.nn.fit(
                x_train,
                x_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor="val_loss", patience= self.patience, mode="min")
                ],
            )
            
            ##plot the train loss and val loss
            plt.figure()
            plt.plot(history.history["loss"], label="Training Loss")
            plt.plot(history.history["val_loss"], label="Validation Loss")
            plt.legend()
            plt.show()
            
            ##train predictions and train mae
            train_predictions, train_mae = self.__calculate_pred_and_err(x_train)
            
            ##create a copy of the train for "results"
            self.results = train_predictions
            
            #set error threshold
            self.error_threshold = self.__calculate_threshold(train_mae[-int(len(train_mae) * 0.5):])
            
            self.trained = True
            
    def test(self, x_test):
        """
        @:param x_test: test data. Must be of shape [ samples, timesteps, features]
        
        -apply inferencing
        
        @:returns test_pred,residuals,anomaly_scores vectors 
        """
        
        logging.info("Autoencoder tests!")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_pred, test_err = self.__calculate_pred_and_err(x_test)
            residuals = np.abs(test_pred - x_test)
            #anomaly_scores = self.__calculate_anomaly_score(residuals , self.error_threshold)
            return test_pred,residuals
        
        
    def loss_of_variance(self,x_train):
    
        pca = PCA(n_components=x_train.shape[1])
        train_set_predictions,mae_loss = self.__calculate_pred_and_err(x_train)
        
        loss_of_variance_by_feature = []
        for i in range(x_train.shape[2]):
          
            variance_of_x_train = pca.fit(x_train[:,:,i]).explained_variance_.sum()
            variance_of_DoEx_train = pca.fit(train_set_predictions[:,:,i] ).explained_variance_.sum()
            variance_loss = (variance_of_x_train - variance_of_DoEx_train)
            loss_of_variance_by_feature.append(variance_loss)

      
        return loss_of_variance_by_feature
        


