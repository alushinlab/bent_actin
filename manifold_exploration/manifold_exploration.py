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
import matplotlib.ticker as ticker
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
ax.scatter(iso_4[:,0], iso_4[:,1], iso_4[:,2],s=10)
for rot in range(0,8):
	print(rot*22.5)
	change_curv = get_curvature_contour(rot*22.5,0,actin_parameters,iso_fit,encoded_preds)
	cmap = plt.cm.viridis(np.linspace(0.1,0.9,len(range(0,8)))[rot])
	_=ax.plot(change_curv[:,0], change_curv[:,1], change_curv[:,2], c=cmap, linewidth=2)
	_=ax.scatter(change_curv[:,0], change_curv[:,1], change_curv[:,2], c=cmap, s=60)

#plt.show()
ax.view_init(elev=35, azim=89)
#ax.view_init(elev=12, azim=170)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.zaxis.set_major_locator(ticker.MultipleLocator(0.5))
plt.tight_layout()
plt.show()

plt.savefig('/mnt/data1/Matt/computer_vision/VAE_squiggle/bent_actin/readme_imgs/viridis_curv_contours_3.png', format='png', dpi=1600)


# plot several curves at once
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
	plt.savefig('/mnt/data1/Matt/computer_vision/VAE_squiggle/bent_actin/readme_imgs/%03d.png'%j, format='png', dpi=1600)
	plt.show()





# decode some samples
encoded_input = layers.Input(shape=(128,))
deco = autoencoder.layers[22](encoded_input)
for i in range(23, len(autoencoder.layers)):
	deco = autoencoder.layers[i](deco)

decoder_model = Model(encoded_input, deco)

interps_dir = '/mnt/data1/Matt/computer_vision/VAE_squiggle/synthetic_data/predicted_interps_moreDOF_2/'
no_rot_no_trans = get_isoRotation_and_isoTranslation_idxs(45,90,actin_parameters)
encodings = encoded_preds[no_rot_no_trans.astype(int)][idxs]
for j in range(0, len(encodings)-1):
	I_MAX = 101.0
	for i in range(0,int(I_MAX)):
		interp = encodings[j]*(1-i/(I_MAX-1)) + encodings[j+1]*(i/(I_MAX-1))
		print((1-i/(I_MAX-1),(i/(I_MAX-1))))
		#plt.scatter(interp[0], interp[1]); plt.show()
		decoded_1 = decoder_model.predict(np.expand_dims(encodings[j],axis=0))[0,:,:,0]
		decoded_2 = decoder_model.predict(np.expand_dims(encodings[j+1],axis=0))[0,:,:,0]
		decoded_interp = decoder_model.predict(np.expand_dims(interp,axis=0))[0,:,:,0]
		#fig,ax = plt.subplots(1,3); _=ax[0].imshow(decoded_1);_=ax[1].imshow(decoded_2);_=ax[2].imshow(decoded_interp); _=plt.show()
		
		with mrcfile.new(interps_dir + 'actin_bent%03d%03d.mrc'%(j,i), overwrite=True) as mrc:
			mrc.set_data(decoded_interp.astype('float32'))












check_num = 1
im_to_test = np.expand_dims(np.expand_dims(noise_holder[check_num], axis=0),axis=-1)
encoded_pred_1 = encoder_model.predict(im_to_test)[0]

encoded_pred = encoded_pred_0*0.5 + encoded_pred_1*0.5


decoded_0 = decoder_model.predict(np.expand_dims(encoded_preds[0],axis=0))[0,:,:,0]
plt.imshow(decoded_0); plt.show()







