#!/mnt/data1/Matt/anaconda_install/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
# import of python packages
import numpy as np
import matplotlib.pyplot as plt
import keras
import mrcfile
import random
from tqdm import tqdm
from keras import layers
from keras.models import Model
import tensorflow as tf
################################################################################
################################################################################
# load CDAE model


# check conv-dense autoencoder
check_num = 28
cm = plt.get_cmap('gray')#plt.cm.greens
predict_conv = autoencoder.predict(np.expand_dims(train[check_num], axis=0))[0,:,:,0]
predict_dense = autoencoder_three.predict(np.expand_dims(train[check_num], axis=0))[0,:,:,0]
fig,ax = plt.subplots(2,3); ax[0,0].imshow(train[check_num,:,:,0], cmap=cm); ax[0,1].imshow(target[check_num,:,:,0], cmap=cm); ax[1,0].imshow(predict_conv, cmap=cm);ax[1,1].imshow(predict_dense,cmap=cm);  #plt.show(block=False)

encoder_model = Model(autoencoder_three.input, autoencoder_three.layers[19].output)
encoded_pred = encoder_model.predict(np.expand_dims(train[check_num], axis=0))[0]
ax[0,2].plot(encoded_pred); plt.show()




