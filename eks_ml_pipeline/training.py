import numpy as np
from models import autoencoder_model_dish_5g 

def pod_autoencoder_training(pod_training_tensor, model_path = '/root/CodeCommit/trained_models/pod_autoencoder'):
    
    #Initialize autoencoder model
    pod_autoencoder = autoencoder_model_dish_5g()
    
    #Train model
    pod_autoencoder.train(pod_training_tensor)
    
    #Save model
    pod_autoencoder.save_nn(model_path)
        
    return pod_autoencoder
    


if __name__ == "__main__":
    #load training data
    pod_training_tensor = np.loadtxt('/root/CodeCommit/data/pod_training_tensor.txt').reshape(600,12,3)
    
    pod_autoencoder_training(pod_training_tensor)