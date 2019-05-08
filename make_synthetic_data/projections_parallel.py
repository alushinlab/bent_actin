#!/mnt/data1/Matt/anaconda_install/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
# import of python packages
import numpy as np; import matplotlib.pyplot as plt
from EMAN2 import *; from sparx import *; import mrcfile
import json; import glob
from multiprocessing import Pool
import os; from tqdm import tqdm
################################################################################
################################################################################
noNoise_outputDir = '/mnt/data1/Matt/computer_vision/VAE_squiggle/synthetic_data/test_output_noiseless/'
noise_outputDir = '/mnt/data1/Matt/computer_vision/VAE_squiggle/synthetic_data/test_output_noise/'
# import data
folder = '/mnt/data1/Matt/computer_vision/VAE_squiggle/synthetic_data/test_input_mrc/'
actin_orig = []
for file_name in os.listdir(folder):
	if(file_name[-4:] == '.mrc' and file_name[:10] == 'bent_actin'):
		actin_orig.append(EMData(folder+file_name))

################################################################################
# image sizes
box_len = 512; BL = box_len / 2
# microscope parameters, for SPARX/EMAN2 CTF convolution
cs = 2.7; voltage = 300.0 # mm, kV
apix = 1.03; bfactor = 200.0; ampcontr = 10.0 # Angstroms, A^2, unitless
astigmatism_amplitude = 10.0; astigmatism_angle = 0.0# nm, degrees

################################################################################
def launch_parallel_process(thread_idx):
	index=num_per_proc*thread_idx
	for i in tqdm(range(0,num_per_proc)):
		local_random_state = np.random.RandomState(None)
		# First: randomly pick one of the actin mrc files that were loaded into actin_orig
		r0 = local_random_state.randint(0,len(actin_orig))
		rotated_actin = actin_orig[r0].copy() 
		# Rotation angles: azimuth, alt, phi, then Translations: tx, ty,tz
		r1, r2, r3 = 90,90,int(local_random_state.random_sample()*360)
		r4, r5, r6 = local_random_state.normal(0, 75), local_random_state.normal(0, 75), local_random_state.normal(0, 75)
		t = Transform()
		t.set_params({'type':'eman','az':r1, 'alt':r2, 'phi':r3, 'tx':r4, 'ty':r5, 'tz':r6})
		rotated_actin.transform(t) # apply rotation and translation
		proj_eman = rotated_actin.project('standard',Transform()) # project
		# Add a padding to reduce CTF ringing
		proj_eman_np = EMNumPy.em2numpy(proj_eman)
		proj_eman_np_padded = np.pad(proj_eman_np, 256, 'constant')
		center = proj_eman_np_padded.shape[0]/2
		# Save the target image
		target = proj_eman_np_padded[center-BL:center+BL, center-BL:center+BL]
		target = (target - np.mean(target)) / np.std(target) # normalize
		with mrcfile.new(noNoise_outputDir + 'actin_rotated%05d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
			mrc.set_data(target.astype('float32'))
		
		proj_eman = EMNumPy.numpy2em(proj_eman_np_padded) # convert back to EMAN2 format
		# Create CTF object and convolve CTF with projection
		defocus = local_random_state.uniform(1.6, 4.0) # microns (positive means underfocus)
		ctf = generate_ctf([defocus,cs,voltage, apix,bfactor,ampcontr,astigmatism_amplitude, astigmatism_angle])
		proj_eman_ctf = filt_ctf(proj_eman, ctf)
		# Convert to numpy and crop
		proj_np_ctf = EMNumPy.em2numpy(proj_eman_ctf)
		center = proj_np_ctf.shape[0]/2
		proj_np_crop = proj_np_ctf[center-BL:center+BL, center-BL:center+BL]
		proj_np_crop = (proj_np_crop - np.mean(proj_np_crop)) / np.std(proj_np_crop)
		# Take FFT for noise addition
		fft2D = np.fft.fft2(proj_np_crop)
		rand_var = max(local_random_state.normal(60000000,15000000),0.0) # generate variance of noise 500000,10000
		# define noise to be added to image
		noise = local_random_state.normal(0, rand_var**0.5, (box_len,box_len,2))
		noise = noise[:,:,0] + 1j*noise[:,:,1]
		fft2D_plus_noise = fft2D + noise
		rotated_actin_plus_noise = np.fft.ifftn(fft2D_plus_noise).real
		rotated_actin_plus_noise = (rotated_actin_plus_noise - np.mean(rotated_actin_plus_noise)) / np.std(rotated_actin_plus_noise)
		with mrcfile.new(noise_outputDir + 'actin_rotated%05d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
			mrc.set_data(rotated_actin_plus_noise.astype('float32'))
		
		# Write text file with all random values and which actin model (curvature and phi) was chosen
		params = {'actin_num':r0, 'alpha':r1, 'beta':r2, 'gamma':r3, 'tx':r4, 'ty':r5, 'tz':r6, 'defocus':defocus, 'iteration':i+num_per_proc*thread_idx}
		with open(noise_outputDir+'params_%02d.json'%(num_per_proc*thread_idx), 'a') as fp:
			data_to_write = json.dumps(params)
			fp.write(data_to_write + '\n')

################################################################################
# run in parallel
TOTAL_NUM_TO_MAKE = 20
nProcs = 10
num_per_proc = TOTAL_NUM_TO_MAKE / nProcs
if __name__ == '__main__':
	p=Pool(nProcs)
	p.map(launch_parallel_process, range(0, nProcs))

################################################################################
# Now all files are written, combine all json files into one master json file
read_files = glob.glob(noise_outputDir+'params_*.json')
output_list = []
for f in read_files:
	for line in open(f, 'r'):
		output_list.append(json.loads(line))

#sort the json dictionaries based on iteration number
output_list = sorted(output_list, key=lambda i: i['iteration'])
for line in output_list:
	with open(noise_outputDir+'master_params.json', 'a') as fp:
		data_to_write = json.dumps(line)
		fp.write(data_to_write + '\n')




