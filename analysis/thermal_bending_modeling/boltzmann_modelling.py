#!/mnt/data1/Matt/anaconda_install/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
# import of python packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
from tqdm import tqdm
################################################################################

curv1 = np.loadtxt('curvMeasures_ADP.csv', delimiter=',', dtype=str)[:,1].astype('float32')
curv2 = np.loadtxt('curvMeasures_ADP_Pi.csv', delimiter=',', dtype=str)[:,1].astype('float32')


fig, ax = plt.subplots(2,2)
_=ax[0,0].hist(curv1, bins=np.arange(0,0.015,0.0002),color='#4682B4', ec='#4682B4', linewidth=0, alpha=0.5,density=True)
_=ax[0,0].hist(curv2, bins=np.arange(0,0.015,0.0002),color='#ff870f', ec='#ff870f', linewidth=0, alpha=0.5,density=True)
_=ax[0,0].set_xlim(0,0.008)

_=ax[0,1].hist(curv2, bins=np.arange(0,0.015,0.0002),color='#ff870f', ec='#ff870f', linewidth=0, alpha=0.5,density=True)
_=ax[0,1].hist(curv1, bins=np.arange(0,0.015,0.0002),color='#4682B4', ec='#4682B4', linewidth=0, alpha=0.5,density=True)
_=ax[0,1].set_xlim(0,0.008)

_=ax[1,0].hist(curv1, bins=np.arange(0,0.015,0.0002),color='#4682B4', ec='black',linewidth=0.5,alpha=0.5,density=True)
_=ax[1,0].hist(curv1, bins=np.arange(0,0.015,0.0002),color='#4682B4', ec='black',linewidth=0.5,alpha=0.5,density=True)
_=ax[1,0].hist(curv2, bins=np.arange(0,0.015,0.0002),color='#ff870f', ec='black',linewidth=0.5,alpha=0.5,density=True)
_=ax[1,0].hist(curv2, bins=np.arange(0,0.015,0.0002),color='#ff870f', ec='black',linewidth=0.5,alpha=0.5,density=True)
_=ax[1,0].set_xlim(0,0.008)

_=ax[1,1].hist(curv1, bins=np.arange(0,0.015,0.0002),color='#4682B4', ec='black',linewidth=0.5,alpha=0.5,density=True)
_=ax[1,1].hist(curv2, bins=np.arange(0,0.015,0.0002),color='#ff870f', ec='black',linewidth=0.5,alpha=0.5,density=True)
_=ax[1,1].set_xlim(0,0.008)

plt.show()


fig, ax = plt.subplots(1)
fig.set_size_inches(3,3)
_=ax.hist(curv1, bins=np.arange(0,0.015,0.0002),color='#4682B4', ec='black',linewidth=0.5,alpha=0.6,density=True)
_=ax.hist(curv2, bins=np.arange(0,0.015,0.0002),color='#ff870f', ec='black',linewidth=0.5,alpha=0.6,density=True)
_=ax.set_xlim(0,0.008)

plt.show()

fig, ax = plt.subplots(1)
fig.set_size_inches(3,3)
_=ax.hist(curv1*1000, bins=np.arange(0,15,0.2),color='#4682B4', ec='black',linewidth=0.5,alpha=0.6,density=True)
_=ax.hist(curv2*1000, bins=np.arange(0,15,0.2),color='#ff870f', ec='black',linewidth=0.5,alpha=0.6,density=True)
_=ax.set_xlim(0,8)

plt.show()


################################################################################
################################################################################
############################### Estimate Energies ##############################
################################################################################
################################################################################
from scipy import integrate 

def p_of_kappa(Lp, L, kappa):
	return np.exp(-0.5*Lp*L*kappa*kappa)

def p_of_kappa_2(Lp, L, kappa, a):
	return np.exp(a*-0.5*Lp*L*kappa*kappa)

x_axis = np.linspace(0, 8, 100)
y_axis = p_of_kappa(9, 0.0500, x_axis)
y_axis2 = p_of_kappa(11, 0.0500, x_axis)
y_axis_norm = 1.0/integrate.simps(y_axis, x_axis)
y_axis_2_norm = 1.0/integrate.simps(y_axis2, x_axis)


# Plot just the functions
fig, ax = plt.subplots(1)
fig.set_size_inches(3,3)
_=ax.set_xlim(0,8)
_=ax.set_ylim(0,0.6)
plt.plot(x_axis, y_axis_norm*y_axis,color='#4682B4')
plt.plot(x_axis, y_axis_2_norm*y_axis2,color='#ff870f')
plt.show()

# Plot both functions with both curves
fig, ax = plt.subplots(1)
fig.set_size_inches(3,3)
_=ax.hist(curv1*1000, bins=np.arange(0,15,0.2),color='#4682B4', ec='black',linewidth=0.5,alpha=0.6,density=True)
_=ax.hist(curv2*1000, bins=np.arange(0,15,0.2),color='#ff870f', ec='black',linewidth=0.5,alpha=0.6,density=True)
_=ax.set_xlim(0,8)
_=ax.set_ylim(0,0.6)
plt.plot(x_axis, y_axis_norm*y_axis,color='#4682B4')
plt.plot(x_axis, y_axis_2_norm*y_axis2,color='#ff870f')
plt.show()

# Plot just ADP
fig, ax = plt.subplots(1)
fig.set_size_inches(3,3)
_=ax.hist(curv1*1000, bins=np.arange(0,15,0.2),color='#4682B4', ec='black',linewidth=0.5,alpha=0.6,density=True)
_=ax.set_xlim(0,8)
_=ax.set_ylim(0,0.6)
plt.plot(x_axis, y_axis_norm*y_axis,color='#4682B4')
plt.show()

# Plot just ADP-Pi
fig, ax = plt.subplots(1)
fig.set_size_inches(3,3)
_=ax.hist(curv2*1000, bins=np.arange(0,15,0.2),color='#ff870f', ec='black',linewidth=0.5,alpha=0.6,density=True)
_=ax.set_xlim(0,8)
_=ax.set_ylim(0,0.6)
plt.plot(x_axis, y_axis_2_norm*y_axis2,color='#ff870f')
plt.show()

################################################################################
################################################################################
# Fit alpha value in front for each one for ADP
holder = ax.hist(curv1*1000, bins=np.arange(0,15,0.2),color='#4682B4', ec='black',linewidth=0.5,alpha=0.6,density=True)
data_to_match = [(holder[1][1:] + holder[1][:-1])/2, holder[0]]

def compute_loss(data, alpha_value):
	x_axis = np.linspace(0.0, 8, 100)
	y_axis = p_of_kappa_2(9, 0.0500, x_axis, alpha_value)
	y_axis_norm = 1.0/integrate.simps(y_axis, x_axis)
	
	f = interp1d(x_axis, y_axis_norm*y_axis)
	interpolated = f(data_to_match[0][:40])
	
	difference = data[:40] - interpolated
	return np.sum(difference * difference) / 74

losses = []
for i in range(0, 500):
	loss = compute_loss(data_to_match[1], np.linspace(0.75,0.85,500)[i])
	losses.append(loss)

np.linspace(0.75,0.85,500)[np.argmin(losses)]
plt.plot(losses)
plt.show()

# alpha for ADP is 0.8003
x_axis = np.linspace(0, 8, 100)
y_axis = p_of_kappa_2(9, 0.0500, x_axis, 0.8003)
y_axis_norm = 1.0/integrate.simps(y_axis, x_axis)

fig, ax = plt.subplots(1)
fig.set_size_inches(3,3)
_=ax.hist(curv1*1000, bins=np.arange(0,15,0.2),color='#4682B4', ec='black',linewidth=0.5,alpha=0.6,density=True)
_=ax.set_xlim(0,8)
_=ax.set_ylim(0,0.6)
plt.plot(x_axis, y_axis_norm*y_axis,color='#4682B4')
plt.show()




# Fit alpha value in front for each one for ADP-Pi
holder = ax.hist(curv2*1000, bins=np.arange(0,15,0.2),color='#4682B4', ec='black',linewidth=0.5,alpha=0.6,density=True)
data_to_match = [(holder[1][1:] + holder[1][:-1])/2, holder[0]]

def compute_loss(data, alpha_value):
	x_axis = np.linspace(0.0, 8, 100)
	y_axis = p_of_kappa_2(11, 0.0500, x_axis, alpha_value)
	y_axis_norm = 1.0/integrate.simps(y_axis, x_axis)
	
	f = interp1d(x_axis, y_axis_norm*y_axis)
	interpolated = f(data_to_match[0][:40])
	
	difference = data[:40] - interpolated
	return np.sum(difference * difference) / 74

losses = []
for i in range(0, 500):
	loss = compute_loss(data_to_match[1], np.linspace(0.9,0.94,500)[i])
	losses.append(loss)

np.linspace(0.9,0.94,500)[np.argmin(losses)]
plt.plot(losses)
plt.show()

# alpha for ADP-Pi is 0.9295
x_axis = np.linspace(0, 8, 100)
y_axis = p_of_kappa_2(11, 0.0500, x_axis, 0.9295)
y_axis_norm = 1.0/integrate.simps(y_axis, x_axis)

fig, ax = plt.subplots(1)
fig.set_size_inches(3,3)
_=ax.hist(curv2*1000, bins=np.arange(0,15,0.2),color='#ff870f', ec='black',linewidth=0.5,alpha=0.6,density=True)
_=ax.set_xlim(0,8)
_=ax.set_ylim(0,0.6)
plt.plot(x_axis, y_axis_norm*y_axis,color='#ff870f')
plt.show()

################################################################################
################################################################################
x_axis = np.linspace(0, 8, 100)
y_axis = p_of_kappa(9, 0.0500, x_axis)
y_axis2 = p_of_kappa(11, 0.0500, x_axis)
y_axis_norm = 1.0/integrate.simps(y_axis, x_axis)
y_axis_2_norm = 1.0/integrate.simps(y_axis2, x_axis)


cmap_name = 'RdYlBu'
fig, ax = plt.subplots(1)
fig.set_size_inches(3,3)
#_=ax.hist(curv1*1000, bins=np.arange(0,15,0.2),color='#4682B4', ec='black',linewidth=0.5,alpha=0.6,density=True)
_=ax.set_xlim(0,8)
for i in range(0, 11):
	temp_y_axis = p_of_kappa(np.linspace(5,15,11)[i], 0.0500, x_axis)
	temp_y_axis_norm = 1.0/integrate.simps(temp_y_axis, x_axis)
	plt.plot(x_axis, temp_y_axis_norm*temp_y_axis, color = plt.cm.RdYlBu(np.linspace(0.8,0.4,11))[i])

plt.show()


################################################################################
# Now do residuals
x_axis = np.linspace(0.0, 8, 100)
y_axis = p_of_kappa(9, 0.0500, x_axis)
y_axis2 = p_of_kappa_2(9, 0.0500, x_axis, 0.8003)
y_axis_norm = 1.0/integrate.simps(y_axis, x_axis)
y_axis_2_norm = 1.0/integrate.simps(y_axis2, x_axis)

from scipy.interpolate import interp1d
holder = ax.hist(curv1*1000, bins=np.arange(0,15,0.2),color='#4682B4', ec='black',linewidth=0.5,alpha=0.6,density=True)
data_to_match = [(holder[1][1:] + holder[1][:-1])/2, holder[0]]

f = interp1d(x_axis, y_axis_norm*y_axis)
f_fit = interp1d(x_axis, y_axis_2_norm*y_axis2)
naive_adp = f(data_to_match[0][:40])
fit_adp = f_fit(data_to_match[0][:40])

holder = ax.hist(curv1*1000, bins=np.arange(0,15,0.2),color='#4682B4', ec='black',linewidth=0.5,alpha=0.6,density=True)
data_to_match = [(holder[1][1:] + holder[1][:-1])/2, holder[0]]

ADP_residual_naive = (data_to_match[1][:40] - naive_adp)
ADP_residual_fit = (data_to_match[1][:40] - fit_adp)

# Plot just the functions
fig, ax = plt.subplots(1)
fig.set_size_inches(3,3)
_=ax.set_xlim(0,8)
_=ax.set_ylim(-0.1,0.1)
plt.plot(data_to_match[0][:40], ADP_residual_naive,color='#4682B4')
plt.plot(data_to_match[0][:40], ADP_residual_fit,color='gray')
plt.show()



# Do ADP-Pi
x_axis = np.linspace(0.0, 8, 100)
y_axis = p_of_kappa(11, 0.0500, x_axis)
y_axis2 = p_of_kappa_2(11, 0.0500, x_axis, 0.9295)
y_axis_norm = 1.0/integrate.simps(y_axis, x_axis)
y_axis_2_norm = 1.0/integrate.simps(y_axis2, x_axis)

from scipy.interpolate import interp1d
holder = ax.hist(curv2*1000, bins=np.arange(0,15,0.2),color='#4682B4', ec='black',linewidth=0.5,alpha=0.6,density=True)
data_to_match = [(holder[1][1:] + holder[1][:-1])/2, holder[0]]

f = interp1d(x_axis, y_axis_norm*y_axis)
f_fit = interp1d(x_axis, y_axis_2_norm*y_axis2)
naive_adp_pi = f(data_to_match[0][:40])
fit_adp_pi = f_fit(data_to_match[0][:40])

holder = ax.hist(curv2*1000, bins=np.arange(0,15,0.2),color='#ff870f', ec='black',linewidth=0.5,alpha=0.6,density=True)
data_to_match = [(holder[1][1:] + holder[1][:-1])/2, holder[0]]

ADP_Pi_residual_naive = (data_to_match[1][:40] - naive_adp_pi)
ADP_Pi_residual_fit = (data_to_match[1][:40] - fit_adp_pi)

# Plot just the functions
fig, ax = plt.subplots(1)
fig.set_size_inches(3,3)
_=ax.set_xlim(0,8)
_=ax.set_ylim(-0.1,0.1)
plt.plot(data_to_match[0][:40], ADP_Pi_residual_naive,color='#ff870f')
plt.plot(data_to_match[0][:40], ADP_Pi_residual_fit,color='gray')
plt.show()




