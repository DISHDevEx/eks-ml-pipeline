import numpy as np
import tensorflow as tf
from utilities import write_tensor, read_tensor
from training_input import node_autoencoder_input, pod_autoencoder_input, container_autoencoder_input
from training_input import node_pca_input, pod_pca_input, container_pca_input
from sklearn.preprocessing import StandardScaler

#Set random seed
random.seed(10)

#Select random day
random_day = random.choice(list())

#Select input for data ingestion

#Read raw data

pyspark_node_data = Pyspark_data_ingestion(year = input_year, month = input_month, day = input_day, hour = input_hour, setup = input_setup, filter_column_value ='Node')
err, pyspark_node_df = pyspark_node_data.read()

#To Pandas
input_node_processed_df = input_node_processed_df.toPandas() 



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