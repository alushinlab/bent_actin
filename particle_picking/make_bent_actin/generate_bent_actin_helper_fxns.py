################################################################################
# imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from prody import *
from scipy.misc import derivative
import string
################################################################################
# This function returns the points on a parametrically defined helix. The helix
# parameterization is given by x = r*cos(omega*t + phi) + d1
#										 y = r*sin(omega*t + phi) + d2
#										 z = c*t
# where t is in radians and r,c,phi,omega,d1,d2,d3,radius are the params. 
# t_rad should be a list of integers defining the actin subunit number
def evaluate_curve_pts(params, t_rad):
	return np.array([(params[0]*np.cos(params[3]*t_rad + params[2]*np.pi/180.0)+params[4]), 
						  (params[0]*np.sin(params[3]*t_rad + params[2]*np.pi/180.0)+params[5]), 
						  (params[1]*t_rad+params[6]+t_rad)]).T

################################################################################
# Evaluate where the centroids of the curvy points go
def evaluate_curve_pts_curvy(params, t_rad, radius):
	s = (params[1]*t_rad)
	y_disp = params[0]*np.sin(params[3]*t_rad + params[2]*np.pi/180.0)+ params[5]
	return np.array([(params[0]*np.cos(params[3]*t_rad + params[2]*np.pi/180.0)) + params[4], 
						  -1.0*(-1.0*y_disp*np.cos(s/radius) + radius*(1-np.cos(s/radius))),
						  params[6]+radius*np.sin(s/radius)+y_disp*np.sin(s/radius)]).T

################################################################################
# Functions to compute frenet-serret basis vectors for a simple helix. 
def calc_unit_tangent_vect(params, t_rad):
	tangent_vect = np.array([(-1.0*params[0]*params[3])*np.sin(params[3]*t_rad + params[2]*np.pi/180.0), 
									 (params[0]*params[3])*np.cos(params[3]*t_rad + params[2]*np.pi/180.0), 
									 params[1]*np.ones(len(t_rad))]).T
	for i in range(0, len(tangent_vect)):
		tangent_vect[i] = tangent_vect[i] / np.linalg.norm(tangent_vect[i])
	
	return tangent_vect

def calc_unit_normal_vect(params, t_rad):
	normal_vect = np.array([(-1.0*params[0]*params[3]*params[3])*np.cos(params[3]*t_rad + params[2]*np.pi/180.0), 
									 (-1.0*params[0]*params[3]*params[3])*np.sin(params[3]*t_rad + params[2]*np.pi/180.0), 
									 0.0*np.ones(len(t_rad))]).T
	for i in range(0, len(normal_vect)):
		normal_vect[i] = normal_vect[i] / np.linalg.norm(normal_vect[i])
	
	return normal_vect

def calc_unit_binormal_vect(tangent_vect, normal_vect):
	binormal_vect  = []
	for i in range(0, len(tangent_vect)):
		binormal_vect.append(np.cross(tangent_vect[i],normal_vect[i]))
	
	return np.asarray(binormal_vect)

################################################################################
# Functions for calculating Frenet-Serret bases of curved helix
from scipy.misc import derivative
def x_of_t(t, r,c,phi,omega,d1,d2,d3,radius):
	return r*np.cos(omega*t + phi*np.pi/180.0) + d1

def y_of_t(t, r,c,phi,omega,d1,d2,d3,radius):
	s = (c*t)
	y_disp = r*np.sin(omega*t +phi*np.pi/180.0)+ d2
	return -1.0*(-1.0*y_disp*np.cos(s/radius) + radius*(1-np.cos(s/radius)))

def z_of_t(t, r,c,phi,omega,d1,d2,d3,radius):
	s = (c*t)
	y_disp = r*np.sin(omega*t + phi*np.pi/180.0)+ d2
	return d3+radius*np.sin(s/radius)+y_disp*np.sin(s/radius)

def gammat_of_t(t, r,c,phi,omega,d1,d2,d3,radius):
	return np.array([x_of_t(t, r,c,phi,omega,d1,d2,d3,radius), 
						  y_of_t(t, r,c,phi,omega,d1,d2,d3,radius),
						  z_of_t(t, r,c,phi,omega,d1,d2,d3,radius)]).T

def d_gamma_dt(t, r,c,phi,omega,d1,d2,d3,radius):
	dx_dt = derivative(x_of_t,t,dx=1e-4, order=15, args=(r,c,phi,omega,d1,d2,d3,radius))
	dy_dt = derivative(y_of_t,t,dx=1e-4, order=15, args=(r,c,phi,omega,d1,d2,d3,radius))
	dz_dt = derivative(z_of_t,t,dx=1e-4, order=15, args=(r,c,phi,omega,d1,d2,d3,radius))
	return np.array([dx_dt, dy_dt, dz_dt])

def calc_unit_tangent_vect_curvy(t, r,c,phi,omega,d1,d2,d3,radius):
	f_t = d_gamma_dt(t,r,c,phi,omega,d1,d2,d3,radius)[0] / np.linalg.norm(d_gamma_dt(t,r,c,phi,omega,d1,d2,d3,radius),axis=0)
	g_t = d_gamma_dt(t,r,c,phi,omega,d1,d2,d3,radius)[1] / np.linalg.norm(d_gamma_dt(t,r,c,phi,omega,d1,d2,d3,radius),axis=0)
	h_t = d_gamma_dt(t,r,c,phi,omega,d1,d2,d3,radius)[2] / np.linalg.norm(d_gamma_dt(t,r,c,phi,omega,d1,d2,d3,radius),axis=0)
	return np.array([f_t, g_t, h_t]).T

def T_x(t,r,c,phi,omega,d1,d2,d3,radius):
	return d_gamma_dt(t,r,c,phi,omega,d1,d2,d3,radius)[0] / np.linalg.norm(d_gamma_dt(t,r,c,phi,omega,d1,d2,d3,radius),axis=0)

def T_y(t,r,c,phi,omega,d1,d2,d3,radius):
	return d_gamma_dt(t,r,c,phi,omega,d1,d2,d3,radius)[1] / np.linalg.norm(d_gamma_dt(t,r,c,phi,omega,d1,d2,d3,radius),axis=0)

def T_z(t,r,c,phi,omega,d1,d2,d3,radius):
	return d_gamma_dt(t,r,c,phi,omega,d1,d2,d3,radius)[2] / np.linalg.norm(d_gamma_dt(t,r,c,phi,omega,d1,d2,d3,radius),axis=0)


def d_T_dt(t,r,c,phi,omega,d1,d2,d3,radius):
	df_dt = derivative(T_x,t,dx=1e-4, order=15, args=(r,c,phi,omega,d1,d2,d3,radius))
	dg_dt = derivative(T_y,t,dx=1e-4, order=15, args=(r,c,phi,omega,d1,d2,d3,radius))
	dh_dt = derivative(T_z,t,dx=1e-4, order=15, args=(r,c,phi,omega,d1,d2,d3,radius))
	return np.array([df_dt, dg_dt, dh_dt])

def calc_unit_normal_vect_curvy(t,r,c,phi,omega,d1,d2,d3,radius):
	f_t = d_T_dt(t,r,c,phi,omega,d1,d2,d3,radius)[0] / np.linalg.norm(d_T_dt(t,r,c,phi,omega,d1,d2,d3,radius),axis=0)
	g_t = d_T_dt(t,r,c,phi,omega,d1,d2,d3,radius)[1] / np.linalg.norm(d_T_dt(t,r,c,phi,omega,d1,d2,d3,radius),axis=0)
	h_t = d_T_dt(t,r,c,phi,omega,d1,d2,d3,radius)[2] / np.linalg.norm(d_T_dt(t,r,c,phi,omega,d1,d2,d3,radius),axis=0)
	return np.array([f_t, g_t, h_t]).T

def calc_unit_binormal_vect_curvy(tangent_vect, normal_vect):
	binormal_vect  = []
	for i in range(0, len(tangent_vect)):
		binormal_vect.append(np.cross(tangent_vect[i],normal_vect[i]))
	
	return np.asarray(binormal_vect)

################################################################################
def make_curved_filament(p, T_helix, N_helix, B_helix, T_helix_curved, N_helix_curved, B_helix_curved, pts_helix, pts_helix_curved, num_subunits):
	actin_orig_0 = p.select('chain C').copy()
	actin_newRef_0 = actin_orig_0.copy()
	a_T = T_helix[0]; a_N = N_helix[0]; a_B = B_helix[0]; a_0 = pts_helix[0] 
	M_0 = np.column_stack((a_T,a_N,a_B))
	coords_orig_0 = actin_orig_0.getCoords()
	frenet_serret_frame_actin = []
	for i in range(0, len(coords_orig_0)):
		set_to_zero = coords_orig_0[i] - np.average(actin_orig_0.getCoords(), axis=0)
		frenet_serret_frame_actin.append(np.matmul(M_0.T,set_to_zero))
	
	new_chains = []
	bent_p = 0
	for i in range(0, num_subunits):
		b_T=T_helix_curved[i]; b_N=N_helix_curved[i]; b_B=B_helix_curved[i]; b_0 = pts_helix_curved[i]
		M_1 = np.column_stack((b_T,b_N,b_B))
		coords_new_0 = []
		for j in range(0, len(coords_orig_0)):
			rot = np.matmul(M_1,frenet_serret_frame_actin[j])
			rot_shift = rot + b_0
			coords_new_0.append(rot_shift)
		
		coords_new_0 = np.asarray(coords_new_0)
		temp_chain = actin_orig_0.copy()
		temp_chain.setCoords(coords_new_0)
		if(i/26 == 0):
			temp_chain.setChids(string.ascii_uppercase[i])
		elif(i/26==1):
			temp_chain.setChids(string.ascii_lowercase[i%26])
		elif(i < 62):
			temp_chain.setChids(str(i-52))
		new_chains.append(temp_chain)
		if(bent_p == 0):
			bent_p = temp_chain
		
		else:
			bent_p = bent_p + temp_chain
		
		#showProtein(new_chains[i])
	
	#plt.show()
	return bent_p





