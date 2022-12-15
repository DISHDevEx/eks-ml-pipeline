from sklearn.decomposition import PCA
def loss_of_variance(data,data_predictions):
    
    
    """
    Input:
    -----
    data: data data of shape (samples,timesteps,features)
   
    
    data_predictions: decoded(encode(data)) of shape (samples,timesteps,features)
    
    Output:
    ------
    loss_of_variance_by_feature : list
    list of length 3, varaince loss of each feature 
    """
    
   
    pca = PCA(n_components=data.shape[1])
         
    loss_of_variance_by_feature = []
    for i in range(data.shape[2]):
           
        variance_of_data = pca.fit(data[:,:,i]).explained_variance_.sum()
        variance_of_DoE_data = pca.fit(data_predictions[:,:,i] ).explained_variance_.sum()
        variance_loss = (variance_of_data - variance_of_DoE_data)
        loss_of_variance_by_feature.append(variance_loss)
 
       
                     
    return loss_of_variance_by_feature