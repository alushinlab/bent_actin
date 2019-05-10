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
import json
from mpl_toolkits.mplot3d import Axes3D
################################################################################
import keras.backend as K
def custom_loss(weights, outputs):
	def contractive_loss(y_pred, y_true):
		lam = 1e-2
		#print(len(autoencoder.layers))
		mse = K.mean(K.square(y_true - y_pred), axis=1)
		W = K.variable(value=weights)  # N x N_hidden
		W = K.transpose(W)  # N_hidden x N
		h = outputs
		dh = h * (1 - h)  # N_batch x N_hidden
		# N_batch x N_hidden * N_hidden x 1 = N_batch x 1
		contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)
		return mse + contractive
	
	return contractive_loss

################################################################################
# method to import synthetic data from files
def import_synth_data(noise_folder, noNoise_folder, box_length, NUM_IMGS_MIN, NUM_IMGS_MAX):
	noise_holder = np.zeros((NUM_IMGS_MAX-NUM_IMGS_MIN,512,512)); noNoise_holder = np.zeros((NUM_IMGS_MAX-NUM_IMGS_MIN,512,512))
	cntr=0
	print('Loading files from ' + noise_folder)
	for i in tqdm(range(NUM_IMGS_MIN, NUM_IMGS_MAX)):
		file_name = 'actin_rotated%05d.mrc'%(i)
		noise_data = None; noNoise_data = None
		with mrcfile.open(noise_folder + file_name) as mrc:
			if(mrc.data.shape == (box_length,box_length)):
				noise_data = mrc.data
		with mrcfile.open(noNoise_folder + file_name) as mrc:
			if(mrc.data.shape == (box_length,box_length)):
				noNoise_data = mrc.data
				
		if(not np.isnan(noise_data).any() and not np.isnan(noNoise_data).any()): #doesn't have a nan
			noise_holder[cntr] = noise_data
			noNoise_holder[cntr] = noNoise_data
			cntr=cntr+1
		
		else: # i.e. if mrc.data does have an nan, skip it and print a statement
			print('Training image number %d has at least one nan value. Skipping this image.'%i)
	
	return noise_holder[:cntr], noNoise_holder[:cntr]

################################################################################
# load CDAE model
#model_path = '../train_neural_network/non_parallel_non_greedy.h5'
model_path = '../train_neural_network/big_dataset_50000_loss1943.h5'
autoencoder = keras.models.load_model(model_path, custom_objects={'contractive_loss':custom_loss(np.zeros((1,1,1)), np.zeros((1,1)))})

# load samples that exist along manifold
noise_holder, noNoise_holder = import_synth_data('output_noise/', 'output_noNoise/', 512, 0, 2240)

# check conv-dense autoencoder
check_num = 1
cm = plt.get_cmap('gray')#plt.cm.greens
im_to_test = np.expand_dims(np.expand_dims(noise_holder[check_num], axis=0),axis=-1)
prediction = autoencoder.predict(im_to_test)[0,:,:,0]
fig,ax = plt.subplots(2,2); ax[0,0].imshow(noise_holder[check_num,:,:], cmap=cm); ax[0,1].imshow(noNoise_holder[check_num,:,:], cmap=cm); ax[1,0].imshow(prediction, cmap=cm)

encoder_model = Model(autoencoder.input, autoencoder.layers[21].output)
encoded_pred = encoder_model.predict(im_to_test)[0]
ax[1,1].plot(encoded_pred); plt.show()

################################################################################
# load json file
param_list = []
for line in open('output_noise/master_params.json','r'):
	param_list.append(json.loads(line))

# get corners
param_list[0]['tx'], param_list[0]['gamma']
param_list[1]['tx'], param_list[1]['gamma']
param_list[2]['tx'], param_list[2]['ty']
param_list[3]['tx'], param_list[3]['ty']

params = []
for i in range(0, len(param_list)):
	params.append([param_list[i]['gamma'], param_list[i]['tx'], param_list[i]['actin_num'], param_list[i]['iteration']])

params = np.asarray(params)
params = params[params[:,-2].argsort()]
first_actin_idxs = params[1*112:2*112][:,-1]
################################################################################
# Go from encoded input to decoded image corresponding to encoding
# first encode an example
encoded_preds = []
for i in tqdm(range(0, len(noise_holder))):
	check_num = i
	if i in(first_actin_idxs):
		im_to_test = np.expand_dims(np.expand_dims(noise_holder[check_num], axis=0),axis=-1)
		encoder_model = Model(autoencoder.input, autoencoder.layers[21].output)
		encoded_preds.append(encoder_model.predict(im_to_test)[0])

encoded_preds = np.asarray(encoded_preds)


from sklearn.decomposition import PCA
pca = PCA(n_components=4)
PCA_2 = pca.fit_transform(encoded_preds)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(PCA_2[:,1], PCA_2[:,2], PCA_2[:,3]); plt.show()

from sklearn.manifold import Isomap
iso = Isomap(n_neighbors=5, n_components=4)
iso_4 = iso.fit_transform(encoded_preds)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(iso_4[:,0], iso_4[:,1], iso_4[:,2]); plt.show()


cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumsum); plt.show()




check_num = 1
im_to_test = np.expand_dims(np.expand_dims(noise_holder[check_num], axis=0),axis=-1)
encoded_pred_1 = encoder_model.predict(im_to_test)[0]

encoded_pred = encoded_pred_0*0.5 + encoded_pred_1*0.5

# then decode an example
encoded_input = layers.Input(shape=(128,))
deco = autoencoder.layers[22](encoded_input)
for i in range(23, len(autoencoder.layers)):
	deco = autoencoder.layers[i](deco)

decoder_model = Model(encoded_input, deco)
decoded_0 = decoder_model.predict(np.expand_dims(encoded_pred,axis=0))[0,:,:,0]
plt.imshow(decoded_0); plt.show()







