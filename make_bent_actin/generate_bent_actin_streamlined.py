#!/mnt/data1/Matt/anaconda_install/anaconda2/envs/tf-gpu/bin/python
################################################################################
import argparse; import sys
parser = argparse.ArgumentParser('Generate a PDB file of a bent actin filament.')
parser.add_argument('--subunits', type=int, help='the number of total subunits for this bent actin filament. MUST BE ODD')
parser.add_argument('--radius', type=float, help='the radius of curvature for this bent actin. Typically a float between 150.0 to 100000.0')
parser.add_argument('--outputFileName', type=str, help='Output file name. If not specified, will default to bent_actin_NN_subunits_RRR_curvature.pdb')
parser.add_argument('--plotting', type=str, help='Set to False if you do not want to see centroids plot. Leave blank or True otherwise')
args = parser.parse_args()

if(args.radius == None or args.subunits == None):
	sys.exit('Please enter both a radius and a subunit number')

if(args.subunits % 2 == 0):
	sys.exit('Please enter an odd number for your --subunits flag')

num_subunits = args.subunits
radius = args.radius
output_file_name = args.outputFileName
if(args.plotting == 'False'): plotting = False
else: plotting = False
print('The program will make an actin filament composed of %0d subunits with a radius of curvature of %0d')%(num_subunits,radius)
################################################################################
# imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from prody import *
from scipy.misc import derivative
from generate_bent_actin_helper_fxns import *
################################################################################
# import the pdb file as the model p
p = parsePDB('./actin_28mer_centered.pdb')
Char2Num = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 
						  8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P',
						  16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W',
						  23:'X', 24:'Y', 25:'Z', 26:'a', 27:'b', 28:'c', 29:'d',
						  30:'e', 31:'f', 32:'g'}
# get chains corresponding to each helix
helix_1_chains = []
for i in range(0, len(Char2Num)):
	helix_1_chains.append(p.select('chain '+Char2Num[i]))

# get the coordinates for each atom of each actin subunit
helix_1_coords = []
for i in range(0,len(helix_1_chains)):
	helix_1_coords.append(helix_1_chains[i].getCoords())

# make each helix into a [num_chains x num_atoms_per_actin x 3] array
helix_1 = np.asarray(helix_1_coords)
helix_1_centroids = np.average(helix_1, axis=1) - 256.0*1.03
helix_1_centroids = np.asarray(sorted(helix_1_centroids, key=lambda k: [k[2]]))

################################################################################
################################################################################
# plot the predicted points (orange) and the centroids. They should match up.
# The helical parameters were determined by fitting using a 7-dimensional gradient
# descent algorithm, function call is in make_bent_actin.py file
r,c,phi,omega,d1,d2,d3 = 15.76719916835962, 1614.757110066779, 170.46340773863406,\
								 -166.69125153876897, 0.13386832018203787,\
								 0.019234057071704225, 21.08761852485995
t = np.linspace(-16,16,33)
t_rad = t * np.pi/180.0
t_curved = np.linspace(-1*(num_subunits/2),num_subunits/2,num_subunits)
t_rad_curved = t_curved * np.pi/180.0
pts_helix = evaluate_curve_pts(np.array([r,c,phi,omega,d1,d2,d3]),t_rad)
pts_helix_curved = evaluate_curve_pts_curvy(np.array([r,c,phi,omega,d1,d2,d3]),t_rad_curved, radius)

if(plotting):
	fig=plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(pts_helix_curved.T[0],pts_helix_curved.T[1],pts_helix_curved.T[2], alpha=0.5)
	ax.set_xlim(-400,400); ax.set_ylim(-400,400); ax.set_zlim(-400,400)
	ax.scatter(pts_helix.T[0],pts_helix.T[1],pts_helix.T[2], alpha=0.5)
	plt.show()

################################################################################
################################################################################
# Once I am satisfied with the curve based on centroids, I can calculate my 
# Frenet-serret bases
T_helix = calc_unit_tangent_vect(np.array([r,c,phi,omega]),t_rad)
N_helix = calc_unit_normal_vect(np.array([r,c,phi,omega]),t_rad)
B_helix = calc_unit_binormal_vect(T_helix, N_helix)

T_helix_curved = calc_unit_tangent_vect_curvy(t_rad_curved, r,c,phi,omega,d1,d2,d3,radius)
N_helix_curved = calc_unit_normal_vect_curvy(t_rad_curved,r,c,phi,omega,d1,d2,d3,radius)
B_helix_curved = calc_unit_binormal_vect_curvy(T_helix_curved, N_helix_curved)

################################################################################
# Bend the actin based on the specified Frenet-serret bases of the straight and 
# curved actin. The straight one is given to convert from cartesian coordinates
# of one of the subunits to TNB coordinates.
bent_p = make_curved_filament(p, T_helix, N_helix, B_helix, T_helix_curved, N_helix_curved, B_helix_curved, pts_helix, pts_helix_curved, num_subunits)

if (output_file_name == None):
	output_file_name = 'bent_actin_%0d_subunits_%0d_curvature.pdb'%(num_subunits,radius)

writePDB(output_file_name, bent_p)
print('Wrote bent actin filament to PDB file: ' + output_file_name + '\nExiting...')




