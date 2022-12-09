import numpy as np
from models import autoencoder_model_dish_5g 

def pod_autoencoder_testing(pod_testing_tensor, model_path = '/root/CodeCommit/trained_models/pod_autoencoder'):
    
    #Load trained model
    pod_autoencoder = tf.keras.models.load_model(model_path)
    
    #Make predictions
    test_predictions = pod_autoencoder.predict(pod_testing_tensor)
    
    #Calculate residuals for testing data == anomaly score
    test_residuals = np.abs(test_predictions - pod_testing_tensor)
    
    return test_predictions, test_residuals
    


if __name__ == "__main__":
    #load testing data
    pod_testing_tensor = np.loadtxt('/root/CodeCommit/data/pod_training_tensor_2.txt').reshape(600,12,3)
    
    test_predictions, test_residuals = pod_autoencoder_testing(pod_testing_tensor)
    
    #save test_predictions and test_residuals in parquette format