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
class actin_params:
	def __init__(self, rotation, tx, curvature_num, image_idx):
		self.rot = rotation
		self.tx = tx
		self.curv = curvature_num
		self.image_idx = image_idx
	
	def return_params(self):
		return self.rot, self.tx,self.curv,self.image_idx

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
#model_path = '../train_neural_network/big_dataset_50000_loss1943.h5'
model_path = './white_noise_20000_loss1645.h5'
autoencoder = keras.models.load_model(model_path, custom_objects={'contractive_loss':custom_loss(np.zeros((1,1,1)), np.zeros((1,1)))})

# load samples that exist along manifold
noise_holder, noNoise_holder = import_synth_data('output_noise_white/', 'output_noNoise_white_lp15/', 512, 0, 2240)

# check conv-dense autoencoder
check_num = 2
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
for line in open('output_noise_white/master_params.json','r'):
	param_list.append(json.loads(line))

params = []
for i in range(0, len(param_list)):
	params.append([param_list[i]['gamma'], param_list[i]['tx'], param_list[i]['actin_num'], param_list[i]['iteration']])

actin_parameters = []
params = np.asarray(params)
for i in range(0, len(params)):
	actin_parameters.append(actin_params(params[i][0], params[i][1], params[i][2], params[i][3]))

################################################################################
# encode all the evenly sampled noisy images
encoded_preds = []
for i in tqdm(range(0, len(noise_holder))):
	check_num = i
	im_to_test = np.expand_dims(np.expand_dims(noise_holder[check_num], axis=0),axis=-1)
	encoder_model = Model(autoencoder.input, autoencoder.layers[21].output)
	encoded_preds.append(encoder_model.predict(im_to_test)[0])

encoded_preds = np.asarray(encoded_preds)

################################################################################
# order the curvatures
curve_orig_order = [-500,-200,125,200,300,500,1000000,-800,-150,100,-300,-125,150,250,400,800,-1000000,-400,-100,-250]
sorted_naive = sorted(curve_orig_order)
curve_correct_order = list(reversed(sorted_naive[:len(sorted_naive)/2])) + list(reversed(sorted_naive[len(sorted_naive)/2:]))
idxs = []
for i in range(0, len(curve_orig_order)):
	idxs.append(curve_orig_order.index(curve_correct_order[i]))

np.asarray(curve_orig_order)[idxs]

################################################################################
# get indices 
actin_parameters = sorted(actin_parameters, key=lambda x:x.return_params()[2])
no_rot_no_trans = []
for i in range(0, len(actin_parameters)):
	param_num = actin_parameters[i].return_params()
	if(param_num[0] == 90 and param_num[1]==0):
		no_rot_no_trans.append(param_num[-1])

no_rot_no_trans = np.asarray(no_rot_no_trans)

################################################################################
# do isometric mapping to go from [2240x128] to [2240x3]
from sklearn.manifold import Isomap
iso = Isomap(n_neighbors=15, n_components=3)
iso_fit = iso.fit(encoded_preds)
iso_4 = iso_fit.transform(encoded_preds)
change_curv = iso_fit.transform(encoded_preds[no_rot_no_trans.astype(int)])[idxs]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(iso_4[:,0], iso_4[:,1], iso_4[:,2])
ax.plot(change_curv[:,0], change_curv[:,1], change_curv[:,2], c='orange', linewidth=3)
ax.scatter(change_curv[:,0], change_curv[:,1], change_curv[:,2], s=75)

ax.plot(change_curv[:,0], change_curv[:,1], change_curv[:,2], c='green', linewidth=3)
ax.scatter(change_curv[:,0], change_curv[:,1], change_curv[:,2], s=75)
plt.show()


cumsum = np.cumsum(pca_fit.explained_variance_ratio_)
plt.plot(cumsum); plt.show()






from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca_fit = pca.fit(encoded_preds)
pca_full = pca_fit.transform(encoded_preds)
pca_subset = pca_fit.transform(encoded_preds[no_rot_no_trans.astype(int)])
plt.scatter(pca_subset[:,0], pca_subset[:,1]); plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_full[:,0], pca_full[:,1], pca_full[:,2])
ax.scatter(pca_subset[:,0], pca_subset[:,1], pca_subset[:,2], s=50)
plt.show()





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
decoded_0 = decoder_model.predict(np.expand_dims(encoded_preds[0],axis=0))[0,:,:,0]
plt.imshow(decoded_0); plt.show()







