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
from helper_manifold_exploration import *
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
cm = plt.get_cmap('afmhot')#plt.cm.greens
im_to_test = np.expand_dims(np.expand_dims(noise_holder[check_num], axis=0),axis=-1)
prediction = autoencoder.predict(im_to_test)[0,:,:,0]
fig,ax = plt.subplots(2,2); ax[0,0].imshow(noise_holder[check_num,:,:], cmap=cm); ax[0,1].imshow(noNoise_holder[check_num,:,:], cmap=cm); ax[1,0].imshow(prediction, cmap=cm)

encoder_model = Model(autoencoder.input, autoencoder.layers[21].output)
encoded_pred = encoder_model.predict(im_to_test)[0]
ax[1,1].plot(encoded_pred); plt.show()

################################################################################
# load json file
actin_parameters = load_json_file('output_noise_white/master_params.json')

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
# do isometric mapping to go from [2240x128] to [2240x3]
from sklearn.manifold import Isomap
iso = Isomap(n_neighbors=15, n_components=3)
iso_fit = iso.fit(encoded_preds)
iso_4 = iso_fit.transform(encoded_preds)

################################################################################
# plot curvature contour lines
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(iso_4[:,0], iso_4[:,1], iso_4[:,2])
for rot in range(0,8):
	print(rot*22.5)
	change_curv = get_curvature_contour(rot*22.5,-90,actin_parameters,iso_fit)
	cmap = plt.cm.viridis(np.linspace(0.1,0.9,len(range(0,8)))[rot])
	_=ax.plot(change_curv[:,0], change_curv[:,1], change_curv[:,2], c=cmap, linewidth=2)
	_=ax.scatter(change_curv[:,0], change_curv[:,1], change_curv[:,2], c=cmap, s=60)

#plt.show()
ax.view_init(elev=35, azim=89)
#ax.view_init(elev=12, azim=170)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.zaxis.set_major_locator(ticker.MultipleLocator(0.5))
plt.show()

plt.savefig('/mnt/data1/Matt/computer_vision/VAE_squiggle/bent_actin/readme_imgs/viridis_curv_contours_3.png', format='png', dpi=50)





for j in range(-90,91,30):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	#ax.scatter(iso_4[:,0], iso_4[:,1], iso_4[:,2])
	for rot in range(0,8):
		#print(rot*22.5)
		change_curv = get_curvature_contour(rot*22.5,j,actin_parameters,iso_fit)
		cmap = plt.cm.viridis(np.linspace(0.1,0.9,len(range(0,8)))[rot])
		_=ax.plot(change_curv[:,0], change_curv[:,1], change_curv[:,2], c=cmap, linewidth=2)
		_=ax.scatter(change_curv[:,0], change_curv[:,1], change_curv[:,2], c=cmap, s=60)
	
	ax.view_init(elev=35, azim=89)
	#ax.view_init(elev=12, azim=170)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
	ax.zaxis.set_major_locator(ticker.MultipleLocator(0.5))
	plt.savefig('/mnt/data1/Matt/computer_vision/VAE_squiggle/bent_actin/readme_imgs/viridis_curv_contours_transx_%03d.png'%j, format='png', dpi=1600)
	plt.show()
















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

cumsum = np.cumsum(pca_fit.explained_variance_ratio_)
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
decoded_0 = decoder_model.predict(np.expand_dims(encoded_preds[0],axis=0))[0,:,:,0]
plt.imshow(decoded_0); plt.show()







