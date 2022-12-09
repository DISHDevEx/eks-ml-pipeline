import numpy as np
from msspackages import get_features
from models import autoencoder_model_dish_5g 

def autoencoder_training(training_tensor, 
                         feature_group_name, 
                         feature_group_created_date, 
                         save_model_path):
    
    features_df = get_features(feature_group_name,feature_group_created_date)
    model_parameters = features_df["model_parameters"].iloc[0]
    
    #Initialize autoencoder model
    autoencoder = autoencoder_model_dish_5g(time_steps=model_parameters["time_steps"], 
                                            batch_size=model_parameters["batch_size"])
    
    #Train model
    autoencoder.fit(training_tensor)
    
    #Save model
    autoencoder.save_nn(save_model_path)
        
    return autoencoder


if __name__ == "__main__":
    #load training data
    pod_training_tensor = np.loadtxt('/root/CodeCommit/data/pod_training_tensor.txt').reshape(600,12,3)
    
    pod_autoencoder = autoencoder_training(pod_training_tensor, 
                                           "pod_autoencoder_ad", 
                                           "11-30-2022", 
                                           '/root/CodeCommit/trained_models/pod_autoencoder')