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
# if you want to see learning curves, plot this
def plot_history(history):
	p1, = plt.plot(history.history['loss']); p2, = plt.plot(history.history['val_loss']); 
	plt.title('Loss'); plt.ylim(ymin=0,ymax=1); plt.legend((p1,p2), ('Training Loss', 'Validation Loss'), loc='upper right', shadow=True)
	plt.show()

################################################################################
# method to import synthetic data from files
def import_synth_data(noise_folder, noNoise_folder, box_length, NUM_IMGS_MIN, NUM_IMGS_MAX):
	noise_holder = np.zeros((NUM_IMGS_MAX-NUM_IMGS_MIN,512,512)); noNoise_holder = np.zeros((NUM_IMGS_MAX-NUM_IMGS_MIN,512,512))
	cntr=0
	print('Loading files from ' + noise_folder)
	for i in tqdm(range(NUM_IMGS_MIN, NUM_IMGS_MAX)):
		file_name = 'actin_rotated%d.mrc'%i
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
#https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
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
folder = '/mnt/data1/Matt/computer_vision/VAE_squiggle/synthetic_data/'
noise_folder = folder + 'rotated_actin_noise_pink_moreNoise_large/'
noNoise_folder_lp = folder + 'rotated_actin_noNoise_pink_moreNoise_large_lp15/'

train, target = import_synth_data(noise_folder, noNoise_folder_lp, 512, 10000, 35000)

#add extra dimension at end because only one color channel
train = np.expand_dims(train, axis=-1)
target = np.expand_dims(target, axis=-1)

FRAC_VAL = int(train.shape[0] * 0.1)
val_train = train[:FRAC_VAL]
val_target = target[:FRAC_VAL]
train = train[FRAC_VAL:]
target = target[FRAC_VAL:]
print('All files loaded and parsed into training and validation sets.')
print('Beginning training')

################################################################################
######### The data should be imported; now build the model #####################
################################################################################
# Define the model
def create_model_dense(training_data, full_training, lr):
	# Instantiate the model
	input_img = layers.Input(shape=(training_data.shape[1:]))
	# Make layers
	x = layers.Conv2D(8, kernel_size=(3,3), padding='same', activation='relu',trainable=full_training)(input_img) #1
	x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x) #2
	x = layers.Dropout(0.0)(x) #3
	x = layers.Conv2D(8, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#4
	x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)#5
	x = layers.Conv2D(8, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#6
	x = layers.MaxPooling2D(pool_size=(1,1), padding='same')(x)#7
	x = layers.Dropout(0.0)(x)#8
	x = layers.Conv2D(8, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#9
	x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)#10
	x = layers.Conv2D(8, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#11
	x = layers.MaxPooling2D(pool_size=(1,1), padding='same')(x)#12
	x = layers.Conv2D(8, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#13
	x = layers.MaxPooling2D(pool_size=(1,1), padding='same')(x)#14
	x = layers.Conv2D(8, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#15
	x = layers.Flatten()(x) #16
	
	#x = DenseLayerAutoencoder([512,128], activation='relu', dropout=0.01)
	
	x = layers.Dense(512, activation='relu')(x) #17
	x = layers.Dropout(0.05)(x) #18
	x = layers.Dense(256, activation='relu')(x) #17
	x = layers.Dropout(0.05)(x) #18
	x = layers.Dense(128, activation='relu')(x) #21
	x = layers.Dropout(0.0)(x) #22
	x = layers.Dense(256, activation='relu')(x)#25
	x = layers.Dropout(0.05)(x)#26
	x = layers.Dense(512, activation='relu')(x)#25
	x = layers.Dropout(0.05)(x)#26
	x = layers.Dense(32768, activation='relu')(x)#27
	x = layers.Reshape((64,64,8))(x)#28
	
	x = layers.UpSampling2D((1,1))(x)#29
	x = layers.Conv2D(8, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#30
	x = layers.UpSampling2D((1,1))(x)#31
	x = layers.Conv2D(8, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#32
	x = layers.UpSampling2D((2,2))(x)#33
	x = layers.Conv2D(8, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#34
	x = layers.UpSampling2D((1,1))(x)#35
	x = layers.Conv2D(8, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#36
	x = layers.UpSampling2D((2,2))(x)#37
	x = layers.Conv2D(8, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#38
	x = layers.UpSampling2D((2,2))(x)#39
	decoded = layers.Conv2D(1, (3,3), activation='linear', padding='same',trainable=full_training)(x)#40
	
	# optimizer
	adam = keras.optimizers.Adam(lr=lr)
	# Compile model
	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer=adam, loss=custom_loss(autoencoder.layers[21].get_weights()[0], autoencoder.layers[21].output), metrics=['mse'])
	autoencoder.summary()
	return autoencoder, x

################################################################################
# Handle model
def train_model(train_data, train_target):
	autoencoder, encoder = create_model_dense(train_data,True, 0.00005)
	es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2, restore_best_weights=True)
	history = autoencoder.fit(x=train_data, y=train_target, epochs=10, batch_size=32, verbose=1, validation_data = (val_train[1:], val_target[1:]), callbacks=[es])
	return [autoencoder, history, encoder]

################################################################################
# train the three models. First do greedy training of outer layers then inner layers
# then train full model
autoencoder_three, history_three, encoder_three = train_model(train,target)

# save the final model
model_save_name = '../saved_autoencoder_models/curr_best_model_large_dataset_018_loss.h5'
print('Model finished training.\nSaving model as ' + model_save_name)
autoencoder_three.save(model_save_name)
plot_history(history)



################################################################################
# check conv autoencoder
check_num = 47
predict_conv = autoencoder.predict(np.expand_dims(train[check_num], axis=0))[0,:,:,0]
fig,ax = plt.subplots(2,3); ax[0,0].imshow(train[check_num,:,:,0]); ax[0,1].imshow(target[check_num,:,:,0]); ax[1,0].imshow(predict_conv);#ax[1,1].imshow(predict_dense);  #plt.show(block=False)
plt.show()

################################################################################
# check dense autoencoder
check_num = 21
predict_conv = autoencoder.predict(np.expand_dims(target[check_num], axis=0))[0,:,:,0]
predict_dense = autoencoder_new.predict(np.expand_dims(target[check_num], axis=0))[0,:,:,0]
fig,ax = plt.subplots(2,3); ax[0,0].imshow(train[check_num,:,:,0]); ax[0,1].imshow(target[check_num,:,:,0]); ax[1,0].imshow(predict_conv);ax[1,1].imshow(predict_dense);  #plt.show(block=False)

encoder_model = Model(autoencoder_new.input, autoencoder_new.layers[21].output)
encoded_pred = encoder_model.predict(np.expand_dims(train[check_num], axis=0))[0]
ax[0,2].plot(encoded_pred); plt.show()

################################################################################
# check conv-dense autoencoder
check_num = 28
cm = plt.get_cmap('gray')#plt.cm.greens
predict_conv = autoencoder.predict(np.expand_dims(train[check_num], axis=0))[0,:,:,0]
predict_dense = autoencoder_three.predict(np.expand_dims(train[check_num], axis=0))[0,:,:,0]
fig,ax = plt.subplots(2,3); ax[0,0].imshow(train[check_num,:,:,0], cmap=cm); ax[0,1].imshow(target[check_num,:,:,0], cmap=cm); ax[1,0].imshow(predict_conv, cmap=cm);ax[1,1].imshow(predict_dense,cmap=cm);  #plt.show(block=False)

encoder_model = Model(autoencoder_three.input, autoencoder_three.layers[19].output)
encoded_pred = encoder_model.predict(np.expand_dims(train[check_num], axis=0))[0]
ax[0,2].plot(encoded_pred); plt.show()


################################################################################
# import 9 bent states and do PCA to see bending
from EMAN2 import *
def import_9bent_states(bent_folder, NUM_IMGS_MAX):
	bent_holder = []
	print('Loading files from ' + bent_folder)
	for i in tqdm(range(0, NUM_IMGS_MAX)):
		file_name = 'actin_bent%d.mrc'%i
		rotated_actin = EMData(bent_folder+file_name)
		BL=256
		t = Transform()
		t.set_params({'type':'eman','az':90, 'alt':90, 'phi':0, 'tx':0, 'ty':0, 'tz':0})
		rotated_actin.transform(t)
		proj_eman = rotated_actin.project('standard',Transform())
		# TODO: Multiply projection by reasonable CTF parameters
		proj_np = EMNumPy.em2numpy(proj_eman)
		center = proj_np.shape[0]/2
		proj_np_crop = proj_np[center-BL:center+BL, center-BL:center+BL]
		proj_np_crop = (proj_np_crop - np.mean(proj_np_crop)) / np.std(proj_np_crop)
		bent_holder.append(proj_np_crop)
	
	return np.asarray(bent_holder)
	

bent_folder = '/mnt/data1/Matt/computer_vision/VAE_squiggle/synthetic_data/bent_actin/'
bent_actins = import_9bent_states(bent_folder, 9)
bent_actins = np.expand_dims(bent_actins, axis=-1)

encoder_model = Model(autoencoder_new.input, autoencoder_new.layers[9].output)
encoder_model.summary()
encoded_bent = []
for i in range(0, len(bent_actins)):
	encoded_bent.append(encoder_model.predict(np.expand_dims(bent_actins[i], axis=0))[0])

encoded_bent = np.asarray(encoded_bent)
for i in range(0, 4):
	plt.plot(encoded_bent[i])

from sklearn.decomposition import PCA
pca = PCA(n_components=64)
PCA_2 = pca.fit_transform(encoded_bent)
plt.plot(PCA_2[:,0], PCA_2[:,1]); plt.show()

interps_dir = '/mnt/data1/Matt/computer_vision/VAE_squiggle/synthetic_data/predicted_interps/'
for j in range(0, len(PCA_2)-1):
	I_MAX = 101.0
	for i in range(0,int(I_MAX)):
		interp = PCA_2[j]*(1-i/(I_MAX-1)) + PCA_2[j+1]*(i/(I_MAX-1))
		print((1-i/(I_MAX-1),(i/(I_MAX-1))))
		#plt.scatter(interp[0], interp[1]); plt.show()
		interp_encoded = pca.inverse_transform(np.expand_dims(interp,axis=0))[0,:]
		decoded_1 = decoder_model.predict(np.expand_dims(encoded_bent[j],axis=0))[0,:,:,0]
		decoded_2 = decoder_model.predict(np.expand_dims(encoded_bent[j+1],axis=0))[0,:,:,0]
		decoded_interp = decoder_model.predict(np.expand_dims(interp_encoded,axis=0))[0,:,:,0]
		#fig,ax = plt.subplots(1,3); _=ax[0].imshow(decoded_1);_=ax[1].imshow(decoded_2);_=ax[2].imshow(decoded_interp); _=plt.show()
		with mrcfile.new(interps_dir + 'actin_bent%03d%03d.mrc'%(j,i), overwrite=True) as mrc:
			mrc.set_data(decoded_interp.astype('float32'))




# Go from encoded input to decoded image corresponding to encoding
encoded_input = layers.Input(shape=(64,))
deco = autoencoder_new.layers[-11](encoded_input)
deco = autoencoder_new.layers[-10](deco)
deco = autoencoder_new.layers[-9](deco)
deco = autoencoder_new.layers[-8](deco)
deco = autoencoder_new.layers[-7](deco)
deco = autoencoder_new.layers[-6](deco)
deco = autoencoder_new.layers[-5](deco)
deco = autoencoder_new.layers[-4](deco)
deco = autoencoder_new.layers[-3](deco)
deco = autoencoder_new.layers[-2](deco)
deco = autoencoder_new.layers[-1](deco)
decoder_model = Model(encoded_input, deco)

decoded_0 = decoder_model.predict(np.expand_dims(encoded_pred_0,axis=0))[0,:,:,0]
plt.imshow(decoded_0); plt.show()
decoded_1 = decoder_model.predict(np.expand_dims(encoded_pred_1,axis=0))[0,:,:,0]
plt.imshow(decoded_1); plt.show()

for i in range(0, len(encoded_bent)):
	decoded_i = decoder_model.predict(np.expand_dims(encoded_bent[i],axis=0))[0,:,:,0]
	plt.imshow(decoded_i); plt.show()



encoded_combo = (0.5*encoded_pred_0 + 0.5*encoded_pred_1)
decoded_combo = decoder_model.predict(np.expand_dims(encoded_combo,axis=0))[0,:,:,0]
plt.imshow(decoded_combo); plt.show()

plt.imshow(train[0,:,:,0]); plt.show()


################################################################################
# use these for loops to plot the filters at each stage
plt.imshow(train[0,:,:,0]);plt.show()
for i in range(0, 8):
	plt.imshow(encoded_pred_0[:,:,i]);plt.show()

encoder_model_earlier = Model(autoencoder.input, autoencoder.layers[5].output)
encoded_pred_0_earlier = encoder_model_earlier.predict(np.expand_dims(train[0], axis=0))[0]
for i in range(0, 8):
	plt.imshow(encoded_pred_0_earlier[:,:,i]);plt.show()

encoder_model_earliest = Model(autoencoder.input, autoencoder.layers[1].output)
encoded_pred_0_earliest = encoder_model_earliest.predict(np.expand_dims(train[0], axis=0))[0]
for i in range(0, 8):
	plt.imshow(encoded_pred_0_earliest[:,:,i]);plt.show()


################################################################################
# predict real data
real_data_dir = '/mnt/data1/Matt/computer_vision/VAE_squiggle/synthetic_data/real_data/'
with mrcfile.open(real_data_dir + 'beta_actin_Au_1157.mrcs') as mrc:
	real_data = mrc.data

# just check conv part of encoder-decoder
for i in range(0, 5):
	check_num = i
	real_img = np.expand_dims(np.expand_dims(real_data[check_num],axis=0),axis=-1)
	cm = plt.cm.gray
	predict_conv = autoencoder.predict(real_img)[0,:,:,0]
	fig,ax = plt.subplots(1,2); ax[0].imshow(real_img[0,:,:,0], cmap=cm); ax[1].imshow(predict_conv, cmap=cm);
	plt.show()
	with mrcfile.new(real_data_dir + 'pred%02d.mrc'%i, overwrite=True) as mrc:
		mrc.set_data(predict_dense)


# check both conv and dense part of encoder-decoder
for i in range(30, 45):
	check_num = i
	real_img = np.expand_dims(np.expand_dims(real_data[check_num],axis=0),axis=-1)
	cm = plt.cm.gray
	predict_conv = autoencoder.predict(real_img)[0,:,:,0]
	predict_dense = autoencoder_three.predict(real_img)[0,:,:,0]
	fig,ax = plt.subplots(2,3); _=ax[0,0].imshow(real_img[0,:,:,0], cmap=cm); _=ax[0,1].imshow(real_img[0,:,:,0], cmap=cm); _=ax[1,0].imshow(predict_conv, cmap=cm);_=ax[1,1].imshow(predict_dense,cmap=cm);  #plt.show(block=False)
	
	encoder_model = Model(autoencoder_three.input, autoencoder_three.layers[19].output)
	encoded_pred = encoder_model.predict(np.expand_dims(train[check_num], axis=0))[0]
	_=ax[0,2].plot(encoded_pred); plt.show()
	with mrcfile.new(real_data_dir + 'pred%02d.mrc'%i, overwrite=True) as mrc:
		mrc.set_data(predict_dense)




