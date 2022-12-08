import numpy as np
from models import autoencoder_model_dish_5g 
#from training_data_builder import pod_training_data_builder


if __name__ == "__main__":
    #build training data
    #pod_training_data, pod_training_tensor, pod_testing_data, pod_testing_tensor = pod_training_model_builder()
    pod_training_tensor = np.loadtxt('/home/sagemaker-user/CodeCommit/data/pod_training_tensor.txt').reshape(600,12,3)
    pod_testing_tensor = np.loadtxt('/home/sagemaker-user/CodeCommit/data/pod_training_tensor_2.txt').reshape(600,12,3)
    
    print(pod_training_tensor)