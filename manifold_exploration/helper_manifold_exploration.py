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
# load json file
def load_json_file(json_file_name):
	param_list = []
	for line in open(json_file_name,'r'):
		param_list.append(json.loads(line))
	
	params = []
	for i in range(0, len(param_list)):
		params.append([param_list[i]['gamma'], param_list[i]['tx'], param_list[i]['actin_num'], param_list[i]['iteration']])
	
	actin_parameters = []
	params = np.asarray(params)
	for i in range(0, len(params)):
		actin_parameters.append(actin_params(params[i][0], params[i][1], params[i][2], params[i][3]))
	
	return actin_parameters

################################################################################
# get those indices of projections of a given rotation and translation
def get_curvature_contour(rot,tx,actin_parameters,iso_fit, encoded_preds,idxs):
	actin_parameters = sorted(actin_parameters, key=lambda x:x.return_params()[2])
	no_rot_no_trans = []
	for i in range(0, len(actin_parameters)):
		param_num = actin_parameters[i]
		if(param_num.rot == rot and param_num.tx==tx):
			no_rot_no_trans.append(param_num.image_idx)
	
	no_rot_no_trans = np.asarray(no_rot_no_trans)
	change_curv = iso_fit.transform(encoded_preds[no_rot_no_trans.astype(int)])[idxs]
	return change_curv



def get_isoRotation_and_isoTranslation_idxs(rot,tx,actin_parameters):
	actin_parameters = sorted(actin_parameters, key=lambda x:x.return_params()[2])
	no_rot_no_trans = []
	for i in range(0, len(actin_parameters)):
		param_num = actin_parameters[i]
		if(param_num.rot == rot and param_num.tx==tx):
			no_rot_no_trans.append(param_num.image_idx)
	
	no_rot_no_trans = np.asarray(no_rot_no_trans)
	return no_rot_no_trans

