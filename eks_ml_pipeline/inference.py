import numpy as np 
import tensorflow as tf
from msspackages import Pyspark_data_ingestion, get_features
from utilities import write_tensor, read_tensor
from training_input import node_autoencoder_input, pod_autoencoder_input, container_autoencoder_input
from training_input import node_pca_input, pod_pca_input, container_pca_input
from sklearn.preprocessing import StandardScaler

#Set random seed
#np.random.seed(10)

raw_data_s3_path = 's3://dish-5g.core.pd.g.dp.eks.logs.e/inference_data/Node/Node_2022_9_11_12.parquet'
#raw_data_s3_path = 's3://dish-5g.core.pd.g.dp.eks.logs.e/inference_data/Container/Container_2022_8_20_9.parquet'
#raw_data_s3_path = 's3://dish-5g.core.pd.g.dp.eks.logs.e/inference_data/Pod/Pod_2022_7_10_20.parquet'

def read_raw_save_processed(raw_data_s3_path):
    
    #Read raw data in parquet format from s3_path
    df = pd.read_parquet(s3_path)
    
    #Read features and parameters
    features_df = get_features(feature_group_name, feature_input_version)
    features = features_df["feature_name"].to_list()
    #remove spaces: that were put by mistake
    features = [feature.strip(' ') for feature in features]
    model_parameters = features_df["model_parameters"].iloc[0]
    time_steps = model_parameters["time_steps"]
    
    #select unique sampling_column (e.g. InstanceId for Node or pod_id for Pod
    random_id = np.random.choice(df[sampling_column].unique())
    print(f'\n***Select data unique {sampling_column} = {random_id}***\n')
    df = df.loc[(df[sampling_column] == random_id)]
    #sort by time
    df = df.sort_values(by='Timestamp').reset_index(drop=True)
        
    #select last time slice of data
    start = df.shape[0] - time_steps
    df = df.loc[start:start+time_steps, scaled_features]
    print("\n*** Inference input data ***")
    print(display(df))
    print("\n***************************************\n")
    
    #scaler transformations
    scaler = StandardScaler()
    scaled_features = ["scaled_" + feature for feature in features]
    df[scaled_features] = scaler.fit_transform(df[features])
    inference_input_tensor = np.expand_dims(df_input, axis = 0)
    

    print("\n***** Inference input tensor shape*****")
    print(inference_input_tensor.shape)
    print("\n*** Inference input tensor ***")
    print(inference_input_tensor)
    print("\n***************************************\n")
    
    write_tensor(tensor = inference_input_tensor, 
                 bucket_name = model_bucketname, 
                 model_name = model_name, 
                 version = model_version, 
                 model_data_type = 'inference_input_tensor')


#def inference(read_input_path, model_path, write_output_path):
    

# def pod_autoencoder_inference(pod_input_data, model_path = '/root/CodeCommit/trained_models/pod_autoencoder'):
    
#     #Convert raw data to tensor 
#     #proprocess
#     #feature enginering
#     pod_testing_tensor = pod_input_data
    
#     #Load trained model
#     pod_autoencoder = tf.keras.models.load_model(model_path)
    
#     #Make predictions
#     predictions = pod_autoencoder.predict(pod_testing_tensor)
    
#     #Calculate residuals == anomaly score
#     residuals = np.abs(predictions - pod_testing_tensor)
    
#     return predictions, residuals


if __name__ == "__main__":
    
    #load raw data
    #pod_testing_tensor = np.loadtxt('/root/CodeCommit/data/pod_training_tensor_2.txt').reshape(600,12,3)
    
    predictions, residuals = pod_autoencoder_inference(pod_testing_tensor)