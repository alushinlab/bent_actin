#!/mnt/data1/Matt/anaconda_install/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
# Import python files
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from prody import *
import string; import sys
################################################################################
#voth 2012; don't use this one, use newer ones
#SD1 = np.concatenate((np.arange(0,29), np.arange(75,143), np.arange(329,345)))
#SD2 = np.concatenate((np.arange(29,35), np.arange(47,65)))
#SD3 = np.concatenate((np.arange(143,175), np.arange(268,329)))
#SD4 = np.concatenate((np.arange(175,215), np.arange(247,258)))

# define subdomain amino acids
# oda 2019 and voth and pollard 2020; use this one
SD1 = np.concatenate((np.arange(0,28), np.arange(65,140), np.arange(333,370)))
SD2 = np.arange(28,65)
SD3 = np.concatenate((np.arange(140,176), np.arange(265,333)))
SD4 = np.arange(176,265)

################################################################################
def load_coords(file_name):
	# Use ProDy to import PDB file
	p = parsePDB(file_name, subset='calpha')
	chids = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']#set(p.getChids())
	chids = chids[::-1]
	print(chids)
	chains = []
	for chain_idx in chids:
		chains.append(p.select('chain ' + chain_idx).copy())
	
	# get the coordinates for each atom of each actin subunit
	coords = []
	for i in range(0,len(chains)):
		coords.append(chains[i].getCoords())
	
	# make each helix into a [num_chains x num_atoms_per_actin x 3] array
	coords = np.asarray(coords)
	coords = coords #- np.average(np.reshape(coords, (14*370,3)), axis=0)
	return coords, np.average(np.reshape(coords, (len(chids)*370,3)), axis=0)

def convert_coords_to_subdomains(coords):
	subdomains = np.zeros((16,4,3))
	for i in range(0, len(coords)):
		subdomains[i][0] = np.average(coords[i][SD1], axis=0)
		subdomains[i][1] = np.average(coords[i][SD2], axis=0)
		subdomains[i][2] = np.average(coords[i][SD3], axis=0)
		subdomains[i][3] = np.average(coords[i][SD4], axis=0)
	
	return subdomains

################################################################################

def three_pt_angle(a,b,c):
	ba = a-b
	bc = c-b
	cosine_ang = np.dot(ba,bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
	angle = np.arccos(cosine_ang)
	return np.degrees(angle)

################################################################################
def compute_dihedral(p):
	p0=p[0]; p1=p[1]; p2=p[2]; p3=p[3]
	b0 = -1.0*(p1-p0)
	b1 = p2-p1
	b2 = p3-p2
	#normalize b1
	b1 = b1 / np.linalg.norm(b1)
	
	# v = projection of b0 onto plane perpendicular to p1 = b0 minus component aligning with b1
	v = b0 - np.dot(b0,b1)*b1
	w = b2 - np.dot(b2,b1)*b1
	x = np.dot(v,w)
	y = np.dot(np.cross(b1,v),w)
	return np.degrees(np.arctan2(y,x))



def compute_dihedral2(p):
	p0=p[0]; p1=p[1]; p2=p[2]; p3=p[3]
	b0 = -1.0*(p1-p0)
	b1 = p2-p1
	b2 = p3-p2
	b0xb1 = np.cross(b0, b1)
	b1xb2 = np.cross(b2, b1)
	b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)
	y = np.dot(b0xb1_x_b1xb2, b1)*(1.0/np.linalg.norm(b1))
	x = np.dot(b0xb1, b1xb2)
	return np.degrees(np.arctan2(y, x))
	

def SD2_coords(coords):
	Dloop = np.arange(35,47)
	SD2s = []
	for i in range(0, len(coords)):
		#SD2s.append(coords[i][np.concatenate((SD1,SD2))])
		SD2s.append(coords[i][Dloop])
	
	return np.asarray(SD2s)











