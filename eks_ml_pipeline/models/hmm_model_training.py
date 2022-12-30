
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from hmmlearn import GaussianHMM
from matplotlib import pyplot as plt

'''
Author: Ruyi Yang
'''

class hmm_model_dish():
    '''
    @: constructor takes in timesteps 
    @: returns object of class
    '''
    
    def __init__(self, train_valid_ratio=0.9, n_components = 9, covariance_type = 'full',n_iter = 1000,time_steps = 12):
        
        # set the super init to make it available for inheritence among child classes
        super().__init__()
        
        # init our model
        self.model = GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter)
        
        #init the validation ratio
        self.train_valid_ratio = train_valid_ratio
        
        #define the threshold
        self.threshold: float = 0.0
        
        #define the state number
        self.n_components = 9
        
        #define the covariance_type
        self,covariance_type = 'fulll'
        
        #define the result df
        self.result_df = None
    
    ## define the training process
     
    def train(self, x_train) -> float:
        '''
        @:param x_train: takes training data into the model fitting
        
        @:returns: validation score, the mse trained on the validation dataset
        '''
        ## model start
        logging.info('Model training start')
        
        tr_df, valid_df = train_test_split(train_df, shuffle=False, train_size=self.train_valid_ratio)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.model.fit(tr_df)

        train_states = self.model.predict(x_train)
        train_samples = [np.random.multivariate_normal(self.model.means_[s], self.model.covars_[s]) \
                 for s in train_states]
        
        #use state to preidtc train_df['values']

        residual_cpu = x_train.T[0]-train_samples.T[0]
        residual_memory = x_train.T[1]-train_samples.T[1]

        absolute_error_cpu = np.abs(residual_cpu)
        absolute_error_memory = np.abs(residual_memory)

        cpu_threshold = np.max(absolute_error_cpu)
        memory_threshold = np.max(absolute_error_memory)
   
        self.threshold =  np.array(np.max(absolute_error_cpu),np.max(absolute_error_cpu))

        self.result_df = x_train.copy()

        
        
        valid_data = valid_df.values()
        valid_states = self.model.predict(valid_data)
        valid_samples = [np.random.multivariate_normal(self.model.means_[s], self.model.covars_[s]) \
                 for s in valid_states]
        
        validation_score = np.power(valid_samples- valid_data, 2).mean()
        
        self.trained = True

        return validation_score
    
    def test(self,x_test):
        """
        @:param x_test: test data
        @:returns test_pred,residuals,anomaly_scores vectors 
        """
        if self.result_df is None:
            print(f'Exception occurred: no train data input')
            
        logging.info("Hidden Markov Model test")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_states = self.model.predict(x_test)
            test_samples = [np.random.multivariate_normal(self.model.means_[s], self.model.covars_[s]) \
                 for s in test_states]
            
            residuals =  x_test.values-test_samples
            abs_error = np.abs(residuals)
            is_anomals = abs_error > self.threshold
            
            temp_df = test_df.copy()
            temp_ df = temp_df.withColumn('predictions',samples)
            
            temp_df = temp_df.withColumn('is_anomaly' ,is_anomaly)
            temp_df = temp_df.withColumn('residuals',abs_error)
            
            
           
            return temp_df
        

        
    

    