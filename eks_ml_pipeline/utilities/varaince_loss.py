
from sklearn.decomposition import PCA
from models import autoencoder_model_dish_5g 
def loss_of_variance(x_train,x_train_predictions):
    
    
    """
    Input:
    -----
    x_train: x_train data of shape (samples,timesteps,features)
   
    
    x_train_predictions: decoded(encode(x_train)) of shape (samples,timesteps,features)
    
    
    Output:
    ------
    loss_of_variance_by_feature : list
    list of length 3, varaince loss of each feature 
    """
    
   
    pca = PCA(n_components=x_train.shape[1])
         
    loss_of_variance_by_feature = []
    for i in range(x_train.shape[2]):
           
        variance_of_x_train = pca.fit(x_train[:,:,i]).explained_variance_.sum()
        variance_of_DoEx_train = pca.fit(x_train_predictions[:,:,i] ).explained_variance_.sum()
        variance_loss = (variance_of_x_train - variance_of_DoEx_train)
        loss_of_variance_by_feature.append(variance_loss)
 
       
    return loss_of_variance_by_feature