from modelling import Autoencoder_Model_Dish_5g
from training_data_builder import pod_training_data_builder

model = Autoencoder_Model_Dish_5g(batch_size = batch_size, time_steps = time_steps)
val_err = model.train(x_train)
    
model.save('pod_model.hdf5')


