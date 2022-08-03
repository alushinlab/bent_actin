#!/mnt/data1/Matt/anaconda_install/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
# Import python files
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
import string; import sys
from scipy.optimize import leastsq
################################################################################
def deriv_axis_spline_curve(x,y,z,res=0.1, order=1):
	cs_x = CubicSpline(np.arange(0,len(x)), x, bc_type='natural')
	x_spline = cs_x(np.arange(-1, len(x), res),order)
	cs_y = CubicSpline(np.arange(0,len(y)), y, bc_type='natural')
	y_spline = cs_y(np.arange(-1, len(y), res),order)
	cs_z = CubicSpline(np.arange(0,len(z)), z, bc_type='natural')
	z_spline = cs_z(np.arange(-1, len(z), res),order)
	central_axis = np.stack((x_spline, y_spline, z_spline),axis=-1)
	return central_axis

file_name = 'ADP_cryodrgn_isolde_frame' + '009'#str(sys.argv[1]).zfill(3)
new_avgs = np.loadtxt('./measured_params/'+file_name+'_centralAxisCoords.txt', delimiter=',')
rise_twist = np.loadtxt('./measured_params/'+file_name+'.csv', delimiter=',')

################################################################################
# Curvature measures
def moving_average(a, n=3) :
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n

# define r'(t)
final_axis_deriv = deriv_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.001)
# define r"(t) for later
final_axis_second_deriv = deriv_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.001, 2)
# define T(t) = r'(t)/||r'(t)||
r_prime_norms = np.repeat(np.expand_dims(np.linalg.norm(final_axis_deriv, axis=-1), axis=-1), 3, axis=-1)
T_of_t = final_axis_deriv / r_prime_norms

# Now, get curvature of the central axis
r_prime = final_axis_deriv.copy()
r_double_prime = final_axis_second_deriv.copy()
k = np.linalg.norm(np.cross(r_prime, r_double_prime), axis=-1) / (np.linalg.norm(r_prime, axis=-1)**3)


roll_window = 2000
k_roll = moving_average(k,roll_window)

def get_k_roll_average(new_avgs, win):
	# define r'(t)
	final_axis_deriv = deriv_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.001)
	# define r"(t) for later
	final_axis_second_deriv = deriv_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.001, 2)
	# define T(t) = r'(t)/||r'(t)||
	r_prime_norms = np.repeat(np.expand_dims(np.linalg.norm(final_axis_deriv, axis=-1), axis=-1), 3, axis=-1)
	T_of_t = final_axis_deriv / r_prime_norms
	
	# Now, get curvature of the central axis
	r_prime = final_axis_deriv.copy()
	r_double_prime = final_axis_second_deriv.copy()
	k = np.linalg.norm(np.cross(r_prime, r_double_prime), axis=-1) / (np.linalg.norm(r_prime, axis=-1)**3)
	
	roll_window = win
	k_roll = moving_average(k,roll_window)
	return k_roll[12000:27000].mean()


################################################################################
################################################################################
# Full plot
'''fig, ax = plt.subplots(3)
fig.set_size_inches(7.5, 10)
ax[0].plot(np.arange(0,40,2),rise_twist[:,1][::2]*1.03)
ax[0].plot(np.arange(1,40,2),rise_twist[:,1][1::2]*1.03)
ax[1].plot(np.arange(0,40,2),rise_twist[:,0][::2]*-1.0)
ax[1].plot(np.arange(1,40,2),rise_twist[:,0][1::2]*-1.0)
ax[2].plot(np.arange(roll_window/2-1000,41000-roll_window/2,1)/1000.0,k_roll[:-1])
ax[0].axvline(x=12, color='black', linestyle='--'); ax[0].axvline(x=27, color='black', linestyle='--'); 
ax[1].axvline(x=12, color='black', linestyle='--'); ax[1].axvline(x=27, color='black', linestyle='--'); 
ax[2].axvline(x=12, color='black', linestyle='--'); ax[2].axvline(x=27, color='black', linestyle='--'); 
ax[1].axhline(y=-166.7, color='black', alpha=0.25)
#ax[0].axhline(y=28.1, color='black', alpha=0.25) # 28.1 A is the rise for the helical ADP
ax[0].axhline(y=27.8, color='black', alpha=0.25) # 27.8 A is the rise for the helical ADP-Pi
ax[0].set_xlim(0,40); ax[1].set_xlim(0,40); ax[2].set_xlim(0,40)
ax[0].set_ylim(25,30); ax[1].set_ylim(-186.7,-146.7); ax[2].set_ylim(0.0,0.00125)
ax[2].set_yticks(np.arange(0.0, 0.00121, 0.0003))
ax[2].set_ylabel('Curvature (A^-1)'); ax[1].set_ylabel('Twist (degrees)'); ax[0].set_ylabel('Rise (A)')
ax[2].set_xlabel('Subunit Index')
plt.show()

# Plot with no overlap
fig, ax = plt.subplots(3)
fig.set_size_inches(4, 6)
#plt.rcParams.update({'font.size': 16})
ax[0].plot(np.arange(0,40,2),rise_twist[:,1][::2]*1.03)
ax[0].plot(np.arange(1,40,2),rise_twist[:,1][1::2]*1.03)
ax[1].plot(np.arange(0,40,2),rise_twist[:,0][::2]*-1.0)
ax[1].plot(np.arange(1,40,2),rise_twist[:,0][1::2]*-1.0)
ax[2].plot(np.arange(roll_window/2-1000,41000-roll_window/2,1)/1000.0,k_roll[:-1], c='tab:green')
ax[1].axhline(y=-166.7, color='black', alpha=0.25)
#ax[0].axhline(y=28.1, color='black', alpha=0.25) # 28.1 A is the rise for the helical ADP
ax[0].axhline(y=27.8, color='black', alpha=0.25) # 27.8 A is the rise for the helical ADP-Pi
ax[0].set_xlim(12,27); ax[1].set_xlim(12,27); ax[2].set_xlim(12,27)
ax[0].set_ylim(25,30); ax[1].set_ylim(-186.7,-146.7); ax[2].set_ylim(0.0,0.00125)
ax[2].set_yticks(np.arange(0.0, 0.00121, 0.0003))
ax[2].set_ylabel('Curvature (A^-1)'); ax[1].set_ylabel('Twist (degrees)'); ax[0].set_ylabel('Rise (A)')
ax[2].set_xlabel('Subunit Index')
ax[2].text(12.5,0.001,'Average Curvature = %s A^-1'%'{0:.5f}'.format(get_k_roll_average(new_avgs,2000)))
#fig.savefig('./threeGraphs_trimmed/'+file_name +'rise_twist_curv.png', bbox_inches='tight',dpi=800)
#plt.show()
'''

################################################################################
################################################################################
################################################################################
twists_pf1 = []
curvs_pf1 = []
rises_pf1 = []
for i in range(0, 10):
#for i in range(3, 10):
	file_name = 'ADP_cryodrgn_isolde_frame' + str(i).zfill(3)
	rise_twist = np.loadtxt('./measured_params/'+file_name+'.csv', delimiter=',')
	new_avgs = np.loadtxt('./measured_params/'+file_name+'_centralAxisCoords.txt', delimiter=',')
	curv_measure = get_k_roll_average(new_avgs, 1000)
	twists_pf1.append(rise_twist[:,0][::2]*-1.0)
	rises_pf1.append(rise_twist[:,1][::2]*1.03)
	curvs_pf1.append(curv_measure*1000)

twists_pf1 = np.asarray(twists_pf1)
rises_pf1 = np.asarray(rises_pf1)
curvs_pf1 =np.expand_dims(np.asarray(curvs_pf1), axis=-1)



# Fit sine wave
from scipy.optimize import curve_fit
def func(X, A,k,omega,B,phi):
	t,curv = X
	return A*(curv)*np.sin(k*curv + omega*t + phi) + B

twists_pf1 = twists_pf1[:,4:16];# rises_pf1 = rises_pf1[:,4:16]
subs = twists_pf1.shape[1]
t = np.linspace(0,2*subs-2,subs) #38
#t = np.linspace(1,2*subs-1,subs) #38
A = 30000/1000.
omega = 0.25
B = -166.7
k = 3000/1000.
phi=np.radians(180)
p0 = A,k,omega,B,phi

A_fit, k_fit, omega_fit, B_fit, phi_fit = curve_fit(func, (np.tile(t,len(twists_pf1)).reshape(len(twists_pf1),subs).ravel(),
								curvs_pf1.repeat(subs).reshape(len(twists_pf1),subs).ravel()), twists_pf1.ravel(),p0)[0]

pcov = curve_fit(func, (np.tile(t,len(twists_pf1)).reshape(len(twists_pf1),subs).ravel(),
								curvs_pf1.repeat(subs).reshape(len(twists_pf1),subs).ravel()), twists_pf1.ravel(),p0)[1]
np.sqrt(np.diag(pcov))

# recreate the fitted curve using the optimized parameters
#data_first_guess = func((t,curv),A,k,omega,B,phi)
frame = 1
fine_t = np.arange(0,max(t),0.1)
data_fit = func((fine_t,curvs_pf1[frame]),A_fit,k_fit,omega_fit,B_fit,phi_fit)

plt.plot(t,  twists_pf1[frame], '.')
plt.plot(fine_t, data_fit)
plt.show()


for i in range(0,10):
	fine_t = np.arange(0,max(t),0.1)
	data_fit = func((fine_t,curvs_pf1[i]),A_fit,k_fit,omega_fit,B_fit,phi_fit)
	#plt.plot(t,  twists_pf1[i], '.')
	plt.plot(fine_t, data_fit)

plt.show()









fit_curv_data_for_export = []
fig, ax = plt.subplots(2,5)
for i in range(0,2):
	for j in range(0,5):
		fine_t = np.arange(0,max(t),0.1)
		data_fit = func((fine_t,curvs_pf1[5*i+j]),A_fit,k_fit,omega_fit,B_fit,phi_fit)
		ax[i,j].plot(t,  twists_pf1[5*i+j], '.')
		ax[i,j].plot(fine_t, data_fit)
		ax[i,j].set_ylim(-150,-180)
		ax[i,j].set_xlim(0,22)
		ax[i,j].text(1,-177,'curvature = %s A^-1'%'{0:.5f}'.format(curvs_pf1[5*i+j][0]))
		fit_curv_data_for_export.append((fine_t,data_fit))

ax[0,0].set_ylabel('Twist (degrees)')
ax[1,0].set_ylabel('Twist (degrees)')
plt.show()


#'GnBu' for pf2; 'RdPu' for pf1
spacing_num = 9
fit_curv_data_for_export_synth = []
fig,ax = plt.subplots(1,2)
cmap_blue = np.linspace(0.10,1,spacing_num)
for i in range(0,spacing_num):
	fine_t = np.arange(0,max(t),0.1)
	data_fit = func((fine_t,np.linspace(0,0.0006,spacing_num)[i]),A_fit,k_fit,omega_fit,B_fit,phi_fit)
	_=ax[0].plot(fine_t, data_fit,c=plt.cm.get_cmap('RdPu')(cmap_blue[i]))
	fit_curv_data_for_export_synth.append((fine_t,data_fit))
	#plt.scatter(fine_t, data_fit,c=cmap_blue[i]/100.0, cmap=plt.cm.get_cmap('RdPu'))

ax[0].set_ylim(-150,-190)
ax[1].imshow(np.expand_dims(cmap_blue,axis=0).T, cmap=plt.cm.get_cmap('RdPu'),aspect=0.5,origin='lower')
ax[1].set_xticks([])
ax[1].set_xticklabels([])
ax[1].set_yticks(np.linspace(0,spacing_num-1,5))
ax[1].set_yticklabels(np.linspace(0,0.0006,5))
ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position('right')
fig.tight_layout()
plt.show()



# save points for graphpad
np.savetxt('ADP_twist_pf1.csv',twists_pf1.T, delimiter=',')
np.savetxt('ADP_twist_pf2.csv',twists_pf1.T, delimiter=',')

fit_curv_data_for_export = np.asarray(fit_curv_data_for_export)
np.savetxt('ADP_fitTwist_yvals_pf1.csv', fit_curv_data_for_export[:,1].T, delimiter=',')
np.savetxt('ADP_fitTwist_yvals_pf2.csv', fit_curv_data_for_export[:,1].T, delimiter=',')


fit_curv_data_for_export_synth = np.asarray(fit_curv_data_for_export_synth)
np.savetxt('ADP_interpTwist_pf1.csv', fit_curv_data_for_export_synth[:,1].T, delimiter=',')
np.savetxt('ADP_interpTwist_pf2.csv', fit_curv_data_for_export_synth[:,1].T, delimiter=',')
np.savetxt('ADP_Pi_interpTwist_pf1.csv', fit_curv_data_for_export_synth[:,1].T, delimiter=',')
np.savetxt('ADP_Pi_interpTwist_pf2.csv', fit_curv_data_for_export_synth[:,1].T, delimiter=',')



np.savetxt('ADP_Pi_twist_pf1.csv',twists_pf1.T, delimiter=',')
np.savetxt('ADP_Pi_twist_pf2.csv',twists_pf1.T, delimiter=',')

fit_curv_data_for_export = np.asarray(fit_curv_data_for_export)
np.savetxt('ADP_Pi_fitTwist_yvals_pf1.csv', fit_curv_data_for_export[:,1].T, delimiter=',')
np.savetxt('ADP_Pi_fitTwist_yvals_pf2.csv', fit_curv_data_for_export[:,1].T, delimiter=',')












