import numpy as np
import tensorflow as tf
from eks_ml_pipeline import autoencoder_model_dish_5g 

def autoencoder_testing(testing_tensor, model_path = '/root/CodeCommit/trained_models/pod_autoencoder'):
    
    #Load trained model
    autoencoder = tf.keras.models.load_model(model_path)
    
    #Make predictions
    test_predictions = autoencoder.predict(testing_tensor)
    
    #Calculate residuals for testing data == anomaly score
    test_residuals = np.abs(test_predictions - testing_tensor)
    
    return test_predictions, test_residuals
    


if __name__ == "__main__":
    #load testing data
    pod_testing_tensor = np.loadtxt('/root/CodeCommit/data/pod_training_tensor_2.txt').reshape(600,12,3)
    
    test_predictions, test_residuals = autoencoder_testing(pod_testing_tensor)
 