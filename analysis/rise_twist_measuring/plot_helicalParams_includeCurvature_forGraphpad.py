#!/mnt/data1/Matt/anaconda_install/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
# Import python files
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
import string; import sys
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

file_name = 'ADP_Pi_cryodrgn_isolde_frame' + str(sys.argv[1]).zfill(3)
#file_name = 'ADP_cryodrgn_isolde_frame' + str(0).zfill(3)
new_avgs = np.loadtxt('./measured_params/'+file_name+'_centralAxisCoords.txt', delimiter=',')
rise_twist = np.loadtxt('./measured_params/'+file_name+'.csv', delimiter=',')

'''

# plot the centroids in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
_=ax.scatter(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], c='green', alpha=0.2, s=300)
_=ax.set_xlim(-200,200); _=ax.set_ylim(-200,200); _=ax.set_zlim(-200,200)
plt.show()
'''
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

'''
plt.plot(np.arange(0,60000,1),k)
plt.plot(np.arange(roll_window/2,60000-roll_window/2+1,1),k_roll)
plt.show()


################################################################################
# Rise/twist stuff
plt.plot(rise_twist[:,1])
plt.show()

fig, ax = plt.subplots(3)
ax[0].plot(np.arange(0,40,2),rise_twist[:,1][::2])
ax[0].plot(np.arange(1,40,2),rise_twist[:,1][1::2])
ax[1].plot(np.arange(0,40,2),rise_twist[:,0][::2])
ax[1].plot(np.arange(1,40,2),rise_twist[:,0][1::2])
ax[2].plot(np.arange(roll_window/2-1000,41000-roll_window/2,1)/1000.0,k_roll[:-1])
plt.show()



rise_twist_curv = np.concatenate((rise_twist, np.expand_dims(k_roll[:-1][::1000],axis=-1)), axis=-1)

color_pf1 = plt.cm.Blues(np.linspace(0.3,1.0,13)) #np.arange(0,13,1)#
color_pf2 = plt.cm.Oranges(np.linspace(0.2,0.6,12)) #np.linspace(0.5,0.7,12)
# plot the centroids in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
_=ax.scatter(rise_twist_curv[:,0][::2][11:-10], rise_twist_curv[:,1][::2][11:-10], rise_twist_curv[:,2][::2][11:-10], c=color_pf1,s=100)
_=ax.plot(rise_twist_curv[:,0][::2][11:-10], rise_twist_curv[:,1][::2][11:-10], rise_twist_curv[:,2][::2][11:-10], c='blue', alpha=0.5)
_=ax.scatter(rise_twist_curv[:,0][1::2][11:-10], rise_twist_curv[:,1][1::2][11:-10], rise_twist_curv[:,2][1::2][11:-10], c=color_pf2, s=100)
_=ax.plot(rise_twist_curv[:,0][1::2][11:-10], rise_twist_curv[:,1][1::2][11:-10], rise_twist_curv[:,2][1::2][11:-10], c='orange', alpha=0.5)
#_=ax.set_xlim(-200,200); _=ax.set_ylim(-200,200); _=ax.set_zlim(-200,200)
plt.show()
'''
rise_twist = np.vstack([
np.arange(-12,28,2),rise_twist[:,1][::2]*1.03,
np.arange(-11,28,2),rise_twist[:,1][1::2]*1.03,
np.arange(-12,28,2),rise_twist[:,0][::2]*-1.0,
np.arange(-11,28,2),rise_twist[:,0][1::2]*-1.0])

curv = (np.arange(roll_window/2-1000,41000-roll_window/2,1)/1000.0-12)[::100],k_roll[:-1][::100]
rise_twist = np.asarray(rise_twist)
curv = np.asarray(curv)
np.savetxt('./forGraphpad/'+file_name + '_riseTwist_graphpad.csv', rise_twist.T, delimiter=',')
np.savetxt('./forGraphpad/'+file_name + '_curv_graphpad.csv', curv.T, delimiter=',')

sys.exit()


# Plot for Greg/Enrique
fig, ax = plt.subplots(3)
fig.set_size_inches(5, 10)
ax[0].plot(np.arange(-12,28,2),rise_twist[:,1][::2]*1.03, marker='o', c='tab:blue')
ax[0].plot(np.arange(-11,28,2),rise_twist[:,1][1::2]*1.03, marker='o', c='blue')
ax[1].plot(np.arange(-12,28,2),rise_twist[:,0][::2]*-1.0, marker='o', c='tab:blue')
ax[1].plot(np.arange(-11,28,2),rise_twist[:,0][1::2]*-1.0, marker='o', c='blue')
ax[2].plot(np.arange(roll_window/2-1000,41000-roll_window/2,1)/1000.0-12,k_roll[:-1], c='tab:blue')
ax[1].axhline(y=-166.7, color='black', alpha=0.25)
ax[0].axhline(y=28.1, color='black', alpha=0.25) # 28.1 A is the rise for the helical ADP
#ax[0].axhline(y=27.8, color='black', alpha=0.25) # 27.8 A is the rise for the helical ADP-Pi
ax[0].set_xlim(0,15); ax[1].set_xlim(0,15); ax[2].set_xlim(0,15)
ax[0].set_ylim(25,30); ax[1].set_ylim(-186.7,-146.7); ax[2].set_ylim(0.0,0.00125)
ax[2].set_yticks(np.arange(0.0, 0.00121, 0.0003))
#fig.savefig('./threeGraphs_forGreg_movie/'+file_name +'rise_twist_curv.png', dpi=800)
plt.show()


sys.exit()


# Plot for Greg/Enrique
fig, ax = plt.subplots(3)
fig.set_size_inches(5, 10)
ax[0].plot(np.arange(-12,28,2),rise_twist[:,1][::2]*1.03, marker='o', c='tab:blue')
ax[0].plot(np.arange(-11,28,2),rise_twist[:,1][1::2]*1.03, marker='o', c='blue')
ax[1].plot(np.arange(-12,28,2),rise_twist[:,0][::2]*-1.0, marker='o', c='tab:blue')
ax[1].plot(np.arange(-11,28,2),rise_twist[:,0][1::2]*-1.0, marker='o', c='blue')
ax[2].plot(np.arange(roll_window/2-1000,41000-roll_window/2,1)/1000.0-12,k_roll[:-1], c='tab:blue')
ax[1].axhline(y=-166.7, color='black', alpha=0.25)
ax[0].axhline(y=28.1, color='black', alpha=0.25) # 28.1 A is the rise for the helical ADP
#ax[0].axhline(y=27.8, color='black', alpha=0.25) # 27.8 A is the rise for the helical ADP-Pi
ax[0].set_xlim(0,15); ax[1].set_xlim(0,15); ax[2].set_xlim(0,15)
ax[0].set_ylim(25,30); ax[1].set_ylim(-186.7,-146.7); ax[2].set_ylim(0.0,0.00125)
ax[2].set_yticks(np.arange(0.0, 0.00121, 0.0003))


ax[0].plot(np.arange(-12,28,2),rise_twist_ADPPi[:,1][::2]*1.03, marker='o', c='tab:orange')
ax[0].plot(np.arange(-11,28,2),rise_twist_ADPPi[:,1][1::2]*1.03, marker='o', c='orange')
ax[1].plot(np.arange(-12,28,2),rise_twist_ADPPi[:,0][::2]*-1.0, marker='o', c='tab:orange')
ax[1].plot(np.arange(-11,28,2),rise_twist_ADPPi[:,0][1::2]*-1.0, marker='o', c='orange')
ax[2].plot(np.arange(roll_window/2-1000,41000-roll_window/2,1)/1000.0-12,k_roll_ADPPi[:-1], c='tab:orange')
ax[1].axhline(y=-166.7, color='black', alpha=0.25)
#ax[0].axhline(y=28.1, color='black', alpha=0.25) # 28.1 A is the rise for the helical ADP
ax[0].axhline(y=27.8, color='black', alpha=0.25) # 27.8 A is the rise for the helical ADP-Pi
ax[0].set_xlim(0,15); ax[1].set_xlim(0,15); ax[2].set_xlim(0,15)
ax[0].set_ylim(25,30); ax[1].set_ylim(-186.7,-146.7); ax[2].set_ylim(0.0,0.00125)
ax[2].set_yticks(np.arange(0.0, 0.00121, 0.0003))
fig.savefig('./threeGraphs_forGreg/'+file_name +'rise_twist_curv.png', dpi=800)
plt.show()




rise_twist_ADPPi = rise_twist.copy()
k_roll_ADPPi = k_roll.copy()



# Plot for ayala
fig, ax = plt.subplots(3)
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
#fig.savefig('./threeGraphs/'+file_name +'rise_twist_curv.svg', dpi=800)
plt.show()






sys.exit()






################################################################################
import pandas as pd
path = './csvs_for_ayala/' + file_name

rise_twist_for_csv = [np.arange(0,67,2),rise_twist[:,1][::2],np.arange(1,67,2),rise_twist[:,1][1::2],
	np.arange(0,67,2),rise_twist[:,0][::2],np.arange(1,67,2),rise_twist[:,0][1::2]]
	
curv_for_csv = [(np.arange(roll_window/2-1000,68000-roll_window/2,1)/1000.0)[::1000],k_roll[:-1][::1000]]

df = pd.DataFrame(rise_twist_for_csv)
df.fillna('', inplace=True)
df.to_csv(path+'rise_twist.csv')

df = pd.DataFrame(curv_for_csv)
df.fillna('', inplace=True)
df.to_csv(path+'curv.csv')













