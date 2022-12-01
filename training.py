import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
import tensorflow as tf
from keras import layers
from matplotlib import pyplot as plt


class Autoencoder_Model_Dish_5g():
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
                 validation_split=.1, epochs=100, nuerons = 128, dropout_rate=.1):
        
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
        
        ##create an x_train
        self.x_train: np.ndarray = None
        
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





    def __calculate_threshold(valid_errors: np.ndarray) -> float:
        return 2 * np.max(valid_errors)



    def __initialize_nn(self, x_train):
        
        ##set x_train
        self.x_train =  x_train
        
        ##actually define the nueral network here. 
        ##Its a Keras sequential with an LSTM, a dropout, a repeat vector, then another LSTM, another dropout, and a timedistributed output. 
        ##The output is mapped back the number of features. 
        ##RepeatVector "Repeats the input n times."https://keras.io/api/layers/reshaping_layers/repeat_vector/
        
        ##none of the nueral network is hard coded (except for the number of nuerons b)
        self.nn = keras.Sequential(
            [
                layers.LSTM(self.nuerons, input_shape=(self.x_train.shape[1], self.x_train.shape[2])),
                layers.Dropout(rate=self.dropout_rate),
                layers.RepeatVector(self.x_train.shape[1]),
                layers.LSTM(self.nuerons, return_sequences=True),
                layers.Dropout(rate=self.dropout_rate),
                layers.TimeDistributed(layers.Dense(self.x_train.shape[2])),
            ]
        )
        
        ##nn compile groups layers into a model; the loss here is MSE, optimizer is Adam, learning rate is predefined
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
    
    
    def train(self, x_train):
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
                self.x_train,
                self.x_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
                ],
            )
            
            ##plot the train loss and val loss
            plt.figure()
            plt.plot(history.history["loss"], label="Training Loss")
            plt.plot(history.history["val_loss"], label="Validation Loss")
            plt.legend()
            plt.show()
            
            ##train predictions and train mae
            train_predictions, train_mae = self.__calculate_pred_and_err(self.x_train)
            
            ##create a copy of the train for "results"
            self.results = train_predictions
            
            #set error threshold
            self.error_threshold = self.__calculate_threshold(train_mae[-int(len(train_mae) * 0.5):])
            print("error_threshold",self.error_threshold)
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
            anomaly_scores = self.__calculate_anomaly_score(residuals , self.error_threshold)
            return test_pred,residuals,anomaly_scores
        
    def customize_matplotlib(self, color="black", labelsize=16, fontsize="xx-large"):
        """
        @:param  color: color you want for your matplot lib (string)
        @:param  labelsize: size you want for your labels (int)
        @:param fontsize: size you want for your font text (int)

        -apply matplot lib changes

        @:returns nothing
        """
        plt.rcParams["xtick.color"] = color
        plt.rcParams["xtick.labelsize"] = labelsize
        plt.rcParams["ytick.color"] = color
        plt.rcParams["ytick.labelsize"] = labelsize
        plt.rcParams["text.color"] = color
        plt.rcParams["axes.labelsize"] = labelsize
        plt.rcParams["axes.labelcolor"] = color
        plt.rcParams["legend.fontsize"] = fontsize
        
    def visualize(self, results_df: pd.DataFrame, metric_name: str, last_train_sample: int = None, title: str = "Anomaly visualization"):
        self.customize_matplotlib()
        PREDICTIONS_COLUMN = "predictions"
        ANOMALIES_COLUMN = "is_anomaly"
        GROUND_TRUTH_COLUMN = "gt_is_anomaly"
        ANOM_SCORE_COLUMN  = "anom_score"
        SCORING_FUNCTION_COLUMN  = "scoring_func"
        ERROR_COLUMN = "error"
        X_LABEL: str = "timestamp"
        
        results_df.index = results_df.index.set_names([X_LABEL])

        fig = plt.figure()
        ax = fig.add_subplot(211)
        fig.set_size_inches(20, 20)
        results_df[metric_name].plot.line(ax=ax)
        columns_labels: list = ["Actual TS"]
            
        if last_train_sample is not None:
            ax.axvspan(results_df.index[0], results_df.index[last_train_sample], alpha=0.5, color="gray")
            columns_labels.append("Training part")

        if PREDICTIONS_COLUMN in results_df and np.any(results_df[PREDICTIONS_COLUMN]):
            results_df.reset_index().plot.scatter(
                x=X_LABEL,
                y=PREDICTIONS_COLUMN,
                ax=ax,
                color="b")
            columns_labels.append("Predictions")

        if ANOMALIES_COLUMN in results_df and np.any(results_df[ANOMALIES_COLUMN]):
            results_df[ANOMALIES_COLUMN] = results_df[ANOMALIES_COLUMN].fillna(False)
            results_df[results_df[ANOMALIES_COLUMN]].reset_index().plot.scatter(
                x=X_LABEL,
                y=metric_name,
                ax=ax,
                color="r")
            columns_labels.append("Anomalies")

        if GROUND_TRUTH_COLUMN in results_df and np.any(results_df[GROUND_TRUTH_COLUMN]):
            results_df[results_df[GROUND_TRUTH_COLUMN]].reset_index().plot.scatter(
                x=X_LABEL,
                y="value",
                ax=ax,
                color="g")
            columns_labels.append("GT Anomalies")

        if GROUND_TRUTH_COLUMN in results_df and ANOMALIES_COLUMN in results_df and np.any(
                results_df[results_df[GROUND_TRUTH_COLUMN] & results_df[ANOMALIES_COLUMN]]):
            results_df[results_df[GROUND_TRUTH_COLUMN] & results_df[ANOMALIES_COLUMN]].reset_index().plot.scatter(
                x=X_LABEL,
                y="value",
                ax=ax,
                color=["magenta"])
            columns_labels.append("Predicted & GT Anomalies")

        ax.legend(columns_labels)

        if ANOM_SCORE_COLUMN in results_df:
            ax2 = fig.add_subplot(212)
            results_df[ANOM_SCORE_COLUMN].reset_index().plot(
                x=X_LABEL,
                y=ANOM_SCORE_COLUMN,
                ax=ax2,
                kind="bar")
            ax2.axes.get_xaxis().set_visible(False)
            ax2.axes.set_ylabel("anomaly score")
            
            
            
            
            
            
            

    
    
        
        

