#!/mnt/data1/Matt/anaconda_install/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
# Import python files
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import string
import glob
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
################################################################################
def define_axis_spline_curve(x,y,z,res=0.1):
	cs_x = CubicSpline(np.arange(0,len(x)), x, bc_type='natural')
	x_spline = cs_x(np.arange(-1, len(x), res))
	cs_y = CubicSpline(np.arange(0,len(y)), y, bc_type='natural')
	y_spline = cs_y(np.arange(-1, len(y), res))
	cs_z = CubicSpline(np.arange(0,len(z)), z, bc_type='natural')
	z_spline = cs_z(np.arange(-1, len(z), res))
	central_axis = np.stack((x_spline, y_spline, z_spline),axis=-1)
	return central_axis

################################################################################
file_names_ADP = sorted(glob.glob('./central_axes/ADP_c*'))
file_names_ADP_Pi = sorted(glob.glob('./central_axes/ADP_P*'))

central_axes_ADP, central_axes_ADP_Pi = [],[]
for i in range(0, len(file_names_ADP)):
	new_avgs = np.loadtxt(file_names_ADP[i],delimiter=',')
	central_axis = define_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.01)
	central_axis = central_axis[1300:-1300]
	central_axes_ADP.append(central_axis)

for i in range(0, len(file_names_ADP_Pi)):
	new_avgs = np.loadtxt(file_names_ADP_Pi[i],delimiter=',')
	central_axis = define_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.01)
	central_axis = central_axis[1300:-1300]
	central_axes_ADP_Pi.append(central_axis)

central_axes_ADP = np.asarray(central_axes_ADP)
central_axes_ADP_Pi = np.asarray(central_axes_ADP_Pi)

################################################################################
# Do PCA
pcas_ADP = []
pcas_ADP_Pi = []
for i in range(0, len(central_axes_ADP)):
	pca = PCA(n_components=3)
	pca.fit(central_axes_ADP[i])
	pcas_ADP.append(pca.transform(central_axes_ADP[i]))

for i in range(0, len(central_axes_ADP_Pi)):
	pca = PCA(n_components=3)
	pca.fit(central_axes_ADP_Pi[i])
	pcas_ADP_Pi.append(pca.transform(central_axes_ADP_Pi[i]))

pcas_ADP = np.asarray(pcas_ADP)
pcas_ADP_Pi = np.asarray(pcas_ADP_Pi)


fig, ax = plt.subplots(3)
ax[0].plot(temp[:,0], temp[:,1])
ax[1].plot(temp[:,0], temp[:,2])
ax[2].plot(temp[:,1], temp[:,2])
plt.show()


################################################################################
# Do deviation from straight
def min_dist(pt,line):
	pt_repeat = np.tile(pt, (len(line),1))
	all_dists = np.linalg.norm(pt_repeat - line, axis=-1)
	return np.min(all_dists)

def compute_deviation_fromStraight(fil_axis): #200000
	pca = PCA(n_components=3)
	pca.fit(fil_axis[:200])
	line = np.average(fil_axis[:200], axis=0) + np.tile(np.linspace(-500,900,200000),(3,1)).T*np.tile(pca.components_[0],(200000,1))
	
	'''fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	_=ax.scatter(fil_axis[:,0], fil_axis[:,1], fil_axis[:,2], c='green', alpha=0.02, s=10)
	_=ax.scatter(line[:,0][::2], line[:,1][::2], line[:,2][::2], c='blue', alpha=0.02, s=10)
	
	_=ax.set_xlim(-200,200); _=ax.set_ylim(-200,200); _=ax.set_zlim(-200,200)
	plt.show()
	'''
	dists = []
	for i in range(0, len(fil_axis)):
		# measure distance between point and line
		dists.append(min_dist(fil_axis[i], line))
	
	return np.asarray(dists)

deviations_ADP, deviations_ADP_Pi = [],[]
for i in range(0, len(central_axes_ADP)):
	deviations_ADP.append(compute_deviation_fromStraight(central_axes_ADP[i]))

for i in range(0, len(central_axes_ADP_Pi)):
	deviations_ADP_Pi.append(compute_deviation_fromStraight(central_axes_ADP_Pi[i]))

deviations_ADP = np.asarray(deviations_ADP)
deviations_ADP_Pi = np.asarray(deviations_ADP_Pi)

for i in range(0,len(deviations_ADP)):
	plt.plot(deviations_ADP[i])

plt.show()


for i in range(0,len(deviations_ADP_Pi)):
	plt.plot(deviations_ADP_Pi[i])

plt.show()


# Handle central axis distances
central_fil_steps_ADP = []
central_fil_steps_ADP_Pi = []
for i in range(0, len(central_axes_ADP)):
	the_dists = np.linalg.norm(central_axes_ADP[i][:-1] - central_axes_ADP[i][1:], axis=-1)
	the_dists = np.concatenate([[0],the_dists])
	central_fil_steps_ADP.append(np.cumsum(the_dists))

for i in range(0, len(central_axes_ADP_Pi)):
	the_dists = np.linalg.norm(central_axes_ADP_Pi[i][:-1] - central_axes_ADP_Pi[i][1:], axis=-1)
	the_dists = np.concatenate([[0],the_dists])
	central_fil_steps_ADP_Pi.append(np.cumsum(the_dists))

central_fil_steps_ADP = np.asarray(central_fil_steps_ADP)
central_fil_steps_ADP_Pi = np.asarray(central_fil_steps_ADP_Pi)



################################################################################
# Save data
np.savetxt('./measured_pcas/PC1_ADP.csv', pcas_ADP[:,:,0].T, delimiter=',')
np.savetxt('./measured_pcas/PC2_ADP.csv', pcas_ADP[:,:,1].T, delimiter=',')
np.savetxt('./measured_pcas/PC3_ADP.csv', pcas_ADP[:,:,2].T, delimiter=',')
np.savetxt('./measured_pcas/PC1_ADP_Pi.csv', pcas_ADP_Pi[:,:,0].T, delimiter=',')
np.savetxt('./measured_pcas/PC2_ADP_Pi.csv', pcas_ADP_Pi[:,:,1].T, delimiter=',')
np.savetxt('./measured_pcas/PC3_ADP_Pi.csv', pcas_ADP_Pi[:,:,2].T, delimiter=',')

np.savetxt('./measured_pcas/deviation_from_straightLine_ADP.csv', deviations_ADP.T, delimiter=',')
np.savetxt('./measured_pcas/deviation_from_straightLine_ADP_Pi.csv', deviations_ADP_Pi.T, delimiter=',')

np.savetxt('./measured_pcas/curved_centerAxis_spacing_ADP.csv', central_fil_steps_ADP.T, delimiter=',')
np.savetxt('./measured_pcas/curved_centerAxis_spacing_ADP_Pi.csv', central_fil_steps_ADP_Pi.T, delimiter=',')




#
# measure sumulative distance
np.linalg.norm(central_axes_ADP[0][1:] - central_axes_ADP[0][:-1], axis=-1)[:200].cumsum()





