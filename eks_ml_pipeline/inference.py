import numpy as np
import tensorflow as tf

def pod_autoencoder_inference(pod_input_data, model_path = '/root/CodeCommit/trained_models/pod_autoencoder'):
    
    #Convert raw data to tensor 
    #proprocess
    #feature enginering
    pod_testing_tensor = pod_input_data
    
    #Load trained model
    pod_autoencoder = tf.keras.models.load_model(model_path)
    
    #Make predictions
    predictions = pod_autoencoder.predict(pod_testing_tensor)
    
    #Calculate residuals == anomaly score
    residuals = np.abs(predictions - pod_testing_tensor)
    
    return predictions, residuals


if __name__ == "__main__":
    
    #load raw data
    pod_testing_tensor = np.loadtxt('/root/CodeCommit/data/pod_training_tensor_2.txt').reshape(600,12,3)
    
    predictions, residuals = pod_autoencoder_inference(pod_testing_tensor)