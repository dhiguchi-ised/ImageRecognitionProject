#TensorFlow - Importing the Libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = tf.keras.models.load_model('CNN_Version1_3Classes')
#model.summary()

for layer in model.layers:
    if 'conv' not in layer.name:
        continue
    filter, bias = layer.get_weights()
    print(layer.name , filter.shape)
    # conv2d_XX (3, 3, # of channels, # of filters) -> output format

# retrieve weights from third layer
filters , bias = model.layers[2].get_weights() # 2 here represents third layer

# normalize the filter values to range 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

f = filters[:,:,:,63] # number here represents numberth filter
fig = plt.figure(figsize=(8,8))
plt.imshow(f[:,:,5],cmap='gray') # number here represents numberth channel
plt.xticks([])
plt.yticks([])
#plot the filters 
plt.show()