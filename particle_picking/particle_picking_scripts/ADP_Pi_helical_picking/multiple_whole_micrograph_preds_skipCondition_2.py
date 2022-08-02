#!/mnt/data1/Matt/anaconda_install/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
print('Loading python packages...')
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
from EMAN2 import *
from scipy import interpolate; from scipy.ndimage import filters
from skimage.morphology import skeletonize_3d; import scipy
import keras.backend as K
from helper_multiple_whole_micrograph_preds import *
import glob
import os; import sys
import signal
################################################################################
INDEX = int(sys.argv[1])
TOTAL_GPUS = int(sys.argv[2])
################################################################################
print('Python packages loaded. Setting CUDA environment...')
os.environ["CUDA_VISIBLE_DEVICES"]= str(INDEX-1)#"2"
################################################################################
################################################################################
# load trained Fully Convolutional Network for semantic segmentation
################################################################################
print('Loading neural network models'); print('')
model_path = './trained_networks/FCN_semantic_segmentation_3.h5'
FCN = keras.models.load_model(model_path)
model_path = './trained_networks/800000training_CCC9887.h5'
autoencoder_three = keras.models.load_model(model_path, custom_objects={'CCC':CCC})
################################################################################
# Special stuff to handle time outs
class TimeoutException(Exception): pass

def timeout_handler(signum, frame):	raise TimeoutException

# Makes it so that when SIGALRM signal sent, it calls the function timeout_handler, which raises your exception
signal.signal(signal.SIGALRM, timeout_handler)

################################################################################
# Define picking function
def run_pick_on_micrograph(file_name):
	################################################################################
	# Load real micrographs
	real_data_dir = ''#'whole_micrographs/'
	big_micrograph_name = file_names[i]#'beta_actin_Au_0757_noDW_bin4.mrc'
	with mrcfile.open(real_data_dir + big_micrograph_name) as mrc:
		real_data = mrc.data
	
	################################################################################
	# Divide up the whole micrograph to feed to the FCN for semantic segmentation
	increment = 32
	extractions = slice_up_micrograph(real_data, increment, hist_matcher)
	preds = FCN.predict(np.expand_dims(extractions, axis=-1))[:,:,:,1]
	stitch_back = stitch_back_seg(real_data.shape, preds, increment)
	
	################################################################################
	################################################################################
	# define actin filaments for whole micrograph. 
	triple_pt_box_size = 20 # orig 48
	num_to_prune = 6 # orig 8
	skel_pruned = skeletonize_and_prune_nubs(stitch_back, triple_pt_box_size, num_to_prune)
	# Now plot end points to see results
	E_img = filters.convolve(skel_pruned,[[1,1,1],[1,10,1],[1,1,1]])
	end_pts = np.asarray(np.argwhere(E_img==11))
	################################################################################
	# Now that we have a clean, image-wide skeleton; do instance segmentation
	filaments = define_filaments_from_skelPruned(skel_pruned, end_pts, 40) #orig 50
	# Plot whole micrograph with segmented pixels
	real_data = -1.0*real_data
	real_data = (real_data - np.mean(real_data)) / np.std(real_data)
	_=plt.imshow(-1.0*real_data[:,25:-25], cmap=plt.cm.gray)
	colors = iter(plt.cm.rainbow(np.linspace(0,1,len(filaments))))
	rs = []; curvatures = []; psi_priors = []
	for j in range(0, len(filaments)):
		_=plt.scatter(filaments[j][:,1]-25, filaments[j][:,0], s=0.1, alpha=0.2, color=next(colors))
		curvature, r, psis = spline_curv_estimate(filaments[j], len(filaments[j]), 20, 1, 40) # sampling, buff, min_threshold #10, 3, 50
		if(curvature != [] and len(curvature) >= 2):
			#r_curv_range = r[np.argwhere(np.logical_and(10<curvature, curvature<1200))][:,0,:]
			r_curv_range = r[np.argwhere(np.logical_and(10<curvature, curvature<1e20))][:,0,:]
			kept_curvs = curvature[np.argwhere(np.logical_and(10<curvature, curvature<1e20))]
			plt.scatter(r_curv_range[:,1]-25, r_curv_range[:,0], c='red',s=5, alpha=0.5)
			rs.append(np.concatenate((r_curv_range, kept_curvs), axis=-1)); curvatures.append(curvature)
			psi_priors.append(psis)
	
	plt.tight_layout()
	plt.savefig('pngs/'+big_micrograph_name[:-4].split('/')[-1]+'.png', dpi=600)
	plt.clf()
	################################################################################
	# Extract and predict rotation
	step_size_px = 1
	rs_full = []; extractions_full = []
	for j in range(0, len(rs)):
		if(rs[j].shape[0] != 0 and rs[j].shape[0]>1):
			curv_rot = np.zeros((rs[j].shape[0], 2))
			xycurvrot = np.concatenate((rs[j], curv_rot), axis=1)
			rs_full.append(xycurvrot)
	
	#rs_full: [:,0]=x_trans; [:,1] = y_trans; [:,2] = rad_of_curv; [:,3] = rotation
	for j in range(0, len(rs_full)):
		rs_full[j][:,3] = psi_priors[j]
	
	################################################################################
	# Prepare star file
	header = '# RELION; version 3.0\n\ndata_\n\nloop_ \n_rlnCoordinateX #1 \n_rlnCoordinateY #2 \n_rlnClassNumber #3 \n_rlnAutopickFigureOfMerit #4 \n_rlnHelicalTubeID #5 \n_rlnAngleTiltPrior #6 \n_rlnAngleRotFlipRatio #7 \n_rlnAnglePsiPrior #8 \n_rlnHelicalTrackLengthAngst #9 \n_rlnAnglePsiFlipRatio #10 \n'
	star_file = header
	for j in range(0, len(rs_full)):
		for k in range(0, len(rs_full[j])):
			if(rs_full[j][k][2] > 99999):
				rs_full[j][k][2] = 99999
			
			star_file = star_file + starify(rs_full[j][k][1]*4.0,rs_full[j][k][0]*4.0,1, rs_full[j][k][2], j+1, 90.0, 0.5, 90+rs_full[j][k][3], k*82.4,0.5)
	
	star_file_name = 'starFiles/'+(big_micrograph_name[:-4].split('/')[-1]).split('_bin4')[0]
	with open(star_file_name+'.star', "w") as text_file:
		text_file.write(star_file)

################################################################################
###################### Define micrographs to pick ##############################
################################################################################
print(''); print('');
print('Neural network models loaded. Starting predictions')
# Load one test image for histogram matching
hist_match_dir = '/mnt/data0/neural_network_training_sets/noise_proj4/'
with mrcfile.open(hist_match_dir + 'actin_rotated%05d.mrc'%1) as mrc:
	hist_matcher = mrc.data

file_names = sorted(glob.glob('../Micrographs_bin4/*.mrc'))#[29:]
file_names = file_names[int(len(file_names)*(1.0/TOTAL_GPUS)*(INDEX-1)):int(len(file_names)*(1.0/TOTAL_GPUS)*INDEX)]
print('Predicting on files from ' + file_names[0] + ' to ' + file_names[-1])

################################################################################
# Do picks!
for i in tqdm(range(0, len(file_names))):
	signal.alarm(10)
	try:
		run_pick_on_micrograph(file_names[i])
	except TimeoutException:
		print('Script hung on micrograph: ' + file_names[i] + ' ...')
		print('Proceeding to next micrograph.')







