#!/mnt/data1/Matt/anaconda_install/anaconda2/envs/matt_EMAN2/bin/python
# Import python files
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from prody import *
from scipy.interpolate import CubicSpline
import string; import sys
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

def deriv_axis_spline_curve(x,y,z,res=0.1, order=1):
	cs_x = CubicSpline(np.arange(0,len(x)), x, bc_type='natural')
	x_spline = cs_x(np.arange(-1, len(x), res),order)
	cs_y = CubicSpline(np.arange(0,len(y)), y, bc_type='natural')
	y_spline = cs_y(np.arange(-1, len(y), res),order)
	cs_z = CubicSpline(np.arange(0,len(z)), z, bc_type='natural')
	z_spline = cs_z(np.arange(-1, len(z), res),order)
	central_axis = np.stack((x_spline, y_spline, z_spline),axis=-1)
	return central_axis

def get_coord_of_min_dist(pt, axis):
	pt_repeat = np.repeat(np.expand_dims(pt, axis=0), len(axis), axis=0)
	dists = np.linalg.norm(pt_repeat - axis, axis=1)
	min_dist_arg = np.argmin(dists)
	vect_to_center = axis[min_dist_arg] - pt
	vect_to_center = vect_to_center / np.linalg.norm(vect_to_center) * 15.76719916835962
	return vect_to_center

# get the argument along the axis spline that corresponds to the subunit
def get_h_of_min_dist(pt, axis):
	pt_repeat = np.repeat(np.expand_dims(pt, axis=0), len(axis), axis=0)
	dists = np.linalg.norm(pt_repeat - axis, axis=1)
	min_dist_arg = np.argmin(dists)
	return min_dist_arg

# Now, try and get phis
def rotate_yaw(alpha, vector):
	R_yaw = np.array([[np.cos(alpha), -1.0*np.sin(alpha), 0],[np.sin(alpha), np.cos(alpha), 0],[0,0,1]])
	rot_vect = np.matmul(R_yaw, vector)
	return rot_vect

def rotate_pitch(beta, vector):
	R_pitch = np.array([[np.cos(beta), 0, np.sin(beta)],[0,1,0],[-1.0*np.sin(beta), 0, np.cos(beta)]])
	rot_vect = np.matmul(R_pitch, vector)
	return rot_vect

def rotate_roll(gamma, vector):
	R_pitch = np.array([[1,0,0],[0,np.cos(gamma), -1.0*np.sin(gamma)],[0,np.sin(gamma), np.cos(gamma)]])
	rot_vect = np.matmul(R_pitch, vector)
	return rot_vect

# Get phis by finding optimal angle based on distance to centroid of rotated vector
def get_phis_optAng(axis_pt, true_pos, rot_matrix):
	vect_to_rotate_to = axis_pt - true_pos#np.matmul(rot_matrix.T, true_pos - axis_pt)
	tang_vect = np.array([15.76719916835962,0,0])#np.matmul(rot_matrix, np.array([15.76719916835962,0,0]))
	best_dist = 10000
	samp_rate = np.radians(90)
	best_alpha = np.radians(0); 
	it_cnt = 0
	# gradient descent to get best phi
	while(best_dist > 0.000001 and samp_rate > np.radians(0.000005) and it_cnt < 100000):
		alpha = best_alpha
		decrease_sample_rate = False
		for i in range(-1,2):
			rotated_vect = rotate_yaw(alpha+i*samp_rate, tang_vect)
			new_vect = np.matmul(rot_matrix, rotated_vect) #+ axis_pt
			temp_dist = np.linalg.norm(new_vect - vect_to_rotate_to)
			if(temp_dist <= best_dist):
				best_dist = temp_dist
				best_alpha = alpha+i*samp_rate
				if(i == 0):
					decrease_sample_rate = True
		if(decrease_sample_rate):
			samp_rate = samp_rate / 2.0
		it_cnt = it_cnt + 1
	print(best_dist, np.linalg.norm(new_vect), np.linalg.norm(vect_to_rotate_to))
	print(vect_to_rotate_to, new_vect)
	return best_alpha, best_dist

def get_R_from_a_to_b(a, b):
	v = np.cross(a,b)
	skew_symm_cp = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
	a_b_dot = np.dot(a,b)
	R = np.eye(3) + skew_symm_cp + np.matmul(skew_symm_cp, skew_symm_cp) * 1.0/(1.0+a_b_dot)
	return R

################################################################################
def matt_parsePDB(file_name):
	with open(file_name, 'r') as f:
		lines = f.readlines()
	
	chains, coords = [] ,[]
	chainNum = 0
	for i in range(0, len(lines)):
		if(lines[i][:3]  == 'TER'):
			chainNum = chainNum + 1
		if(lines[i][:4] == 'ATOM' and lines[i][13:15] == 'CA'):
			chain,x,y,z = lines[i][21] + str(chainNum), float(lines[i][30:38]), float(lines[i][38:46]), float(lines[i][46:54])
			chains.append(chain); coords.append((x,y,z))
			#print lines[i][21] + str(chainNum),
	
	chains = np.asarray(chains)
	coords = np.asarray(coords)
	chain_idxs = list(set(chains))
	coords_per_chain = []
	for i in range(0, len(chain_idxs)):
		coords_per_chain.append(coords[np.argwhere(chains == chain_idxs[i])][:,0])
	
	return np.asarray(coords_per_chain)

################################################################################
################################################################################
# Load in PDB file and get chains
frame_index = 'ADP_Pi_cryodrgn_isolde_frame%s_extended.pdb'%str(sys.argv[1]).zfill(3)
param_file_name = './measured_params/'+frame_index[:-13]+'.csv'
file_name = frame_index#'./stitched_fit_pdbs_3dva/stitched_masterSquigJ38_'+frame_index+'_final_C.pdb'
def load_coords(file_name):
	# make each helix into a [num_chains x num_atoms_per_actin x 3] array
	coords = matt_parsePDB(file_name)
	centroids = np.average(coords, axis=1)
	
	# sort centroids by z
	z_index = np.argsort(centroids[:,2])
	centroids = centroids[z_index]
	return centroids

#centroids_top = load_coords(top_file_name)
#centroids_mid = load_coords(mid_file_name)
#centroids_bot = load_coords(bot_file_name)
centroids =load_coords(file_name)# np.concatenate((centroids_bot, centroids_mid, centroids_top))

centroids = centroids - np.average(centroids, axis=0)
avgs = avgs = (centroids[:-1] + centroids[1:]) / 2.0

plt.plot(np.linalg.norm(centroids[::2][:-1] - centroids[::2][1:], axis=-1), marker='o')
#plt.plot(np.arange(0,67)[::2], np.linalg.norm(centroids[::2][:-1] - centroids[::2][1:], axis=-1), marker='o')
#plt.plot(np.arange(0,67)[1::2], np.linalg.norm(centroids[1::2][:-1] - centroids[1::2][1:], axis=-1), marker='o')
plt.show()

# plot the centroids in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
_=ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], c='green', alpha=0.2, s=300)
_=ax.scatter(avgs[:,0], avgs[:,1], avgs[:,2], c='red')
_=ax.set_xlim(-200,200); _=ax.set_ylim(-200,200); _=ax.set_zlim(-200,200)
plt.show()

################################################################################
#isd_pf1 = np.asarray([np.arange(0,67)[::2], np.linalg.norm(centroids[::2][:-1] - centroids[::2][1:], axis=-1)])
#isd_pf2 = np.asarray([np.arange(0,67)[1::2], np.linalg.norm(centroids[1::2][:-1] - centroids[1::2][1:], axis=-1)])
#
#np.savetxt('intersubunit_distance_pf1.csv', isd_pf1, delimiter=',')
#np.savetxt('intersubunit_distance_pf2.csv', isd_pf2, delimiter=',')
#sys.exit()
#
################################################################################
################################################################################
# Now that the data is loaded in, define an initial axis
orig_axis = define_axis_spline_curve(avgs[:,0], avgs[:,1], avgs[:,2])
center_pointers = np.zeros((centroids.shape))
for i in range(0, len(centroids)):
	center_pointers[i] = get_coord_of_min_dist(centroids[i], orig_axis)

new_avgs = ((center_pointers+centroids)[:-1] + avgs + (center_pointers+centroids)[1:])/3.0

"""fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
_=ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], c='green', alpha=0.2, s=300)
_=ax.scatter(orig_axis[:,0], orig_axis[:,1], orig_axis[:,2], c='red')
_=ax.scatter(avgs[:,0], avgs[:,1], avgs[:,2], c='blue', s=100)
_=ax.scatter(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], c='orange', s=100)
for i in range(0, len(center_pointers)):
	a=np.array([centroids[i], center_pointers[i] + centroids[i]])
	_=plt.plot(a[:,0],a[:,1],a[:,2], c='purple')

_=ax.set_xlim(-200,200); _=ax.set_ylim(-200,200); _=ax.set_zlim(-200,200)
plt.show()
"""
################################################################################
# Iteratively update the central axis estimate
prev_avgs = new_avgs.copy()
for i in tqdm(range(0, 500)):
	temp_axis = define_axis_spline_curve(prev_avgs[:,0], prev_avgs[:,1], prev_avgs[:,2], 0.01)
	center_pointers = np.zeros((centroids.shape))
	for j in range(0, len(centroids)):
		center_pointers[j] = get_coord_of_min_dist(centroids[j], temp_axis)
	new_avgs = ((center_pointers+centroids)[:-1] + prev_avgs + (center_pointers+centroids)[1:])/3.0
	prev_avgs = new_avgs.copy()

# Final axis, for viewing. When doing more calculations, sample more finely
final_axis = define_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.05)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
_=ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], c='green', alpha=0.2, s=300)
_=ax.scatter(final_axis[:,0], final_axis[:,1], final_axis[:,2], c='red')
_=ax.scatter(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], c='orange', s=100)
for i in range(0, len(center_pointers)):
	a=np.array([centroids[i], center_pointers[i] + centroids[i]])
	_=plt.plot(a[:,0],a[:,1],a[:,2], c='purple')

_=ax.set_xlim(-200,200); _=ax.set_ylim(-200,200); _=ax.set_zlim(-200,200)
plt.show()

# Plot, x(t), y(t), z(t)
fig, ax = plt.subplots(1,3)
ax[0].plot(final_axis[:,0])
ax[1].plot(final_axis[:,1])
ax[2].plot(final_axis[:,2])
plt.show()


################################################################################
# get distance between steps along central axis curve
final_axis = define_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.001)
time_steps = np.linalg.norm(final_axis[:-1] - final_axis[1:], axis=1)

# Now that we have the central axis as final_axis, get rise and twist
# Plot h'(t) 
hs = []
for i in range(0, len(centroids)):
	hs.append(get_h_of_min_dist(centroids[i], final_axis))

hs = np.asarray(hs)

from scipy.interpolate import CubicSpline
cs = CubicSpline(np.arange(0,len(hs)), hs, bc_type='natural')
h_of_t = cs(np.arange(0, len(hs), 0.001))[:-1] * time_steps/1.03
h_of_t_prime = cs(np.arange(0, len(hs), 0.001), 1)[:-1] / 1.03 * time_steps 
plt.plot(h_of_t_prime) # resize b/c pixel size is 1.03A/px
plt.scatter(hs, h_of_t_prime[hs])
plt.ylim(26.5,28.5)
plt.show()


################################################################################
################################################################################
# define r'(t)
final_axis_deriv = deriv_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.001)
# define r"(t) for later
final_axis_second_deriv = deriv_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.001, 2)
# define T(t) = r'(t)/||r'(t)||
r_prime_norms = np.repeat(np.expand_dims(np.linalg.norm(final_axis_deriv, axis=-1), axis=-1), 3, axis=-1)
T_of_t = final_axis_deriv / r_prime_norms

rots = []
for i in range(0, len(T_of_t)):
	rots.append(get_R_from_a_to_b(np.array([0,0,1]), T_of_t[i]))

rots = np.asarray(rots)

phis = []; dists = []
for i in range(0, len(centroids)):
	phis.append(get_phis_optAng(final_axis[hs[i]], centroids[i],rots[i*1000])[0])
	dists.append(get_phis_optAng(final_axis[hs[i]], centroids[i],rots[i*1000])[1])

phis = np.asarray(phis)
delta_phis = []
for i in range(0, len(phis)-1):
	delta = (np.degrees(phis[i]) - np.degrees(phis[i+1]))
	if(delta < 0):
		delta = delta +360
	if(delta > 300): # b/c phis wraparound at 360 and they are determined based
		delta = (delta - 360)*-1 #on the gradient descent, I need to get them all on the same phase
	delta_phis.append(delta)

dists = np.asarray(dists)
plt.plot(dists, marker='o')
plt.show()

plt.plot(np.asarray(delta_phis), marker='o')
plt.ylim(145,190)
plt.show()

# Adjust phases for phi
phis_adj = phis.copy()
for i in range(0, len(phis_adj)):
	for j in range(1, len(phis_adj)):
		delta = np.degrees(phis_adj[j-1]) - np.degrees(phis_adj[j])
		if(delta<0):
			phis_adj[j] = phis_adj[j] - 2.0*np.pi
			delta = delta+360

for j in range(1, len(phis_adj)):
	delta = np.degrees(phis_adj[j]) - np.degrees(phis_adj[j-1])
	#if(delta < -180):
	#	diff_from_180 = np.abs(np.abs(delta) - 180)
	#	phis_adj[j] = (phis_adj[j] + 2*np.radians(diff_from_180))

deltas = np.zeros(len(phis_adj-1))
for i in range(1, len(phis_adj)):
	deltas[i] = np.degrees(phis_adj[i]) - np.degrees(phis_adj[i-1])

plt.plot(deltas[1:], marker = 'o')
plt.show()

# Make spline, parameterized by t
phis = phis_adj.copy()
cs = CubicSpline(np.arange(0, len(phis)), phis, bc_type='natural')
phi_of_t = cs(np.arange(0, len(phis), 0.1))
phi_of_t_prime = cs(np.arange(0, len(phis), 0.001),1)[:-1]
plt.plot(np.degrees(phi_of_t_prime))
plt.show()

################################################################################
# Save twist and rise values to external file
twists = np.asarray(delta_phis)[1:]
rises = np.asarray(h_of_t_prime[hs][1:-1])
vals_to_save = np.concatenate((np.expand_dims(twists, axis=-1), np.expand_dims(rises, axis=-1)), axis=-1)
np.savetxt(param_file_name, vals_to_save, delimiter=',')
np.savetxt(param_file_name[:-4] + '_centralAxisCoords.txt', new_avgs, delimiter=',')
np.savetxt(param_file_name[:-4] + '_subunitCentroids.txt', centroids, delimiter=',')
sys.exit()

################################################################################
################################################################################
################################################################################
################################################################################
# Making plots of rise for different filaments
#real_deltas = deltas.copy(); real_h_of_t_prime = h_of_t_prime.copy(); real_hs=hs.copy()
plt.plot(np.arange(59-58,len(real_deltas)-61-58)[::2], real_h_of_t_prime[real_hs][60:-60][::2], marker='o', color='tab:blue')
plt.plot(np.arange(59-58,len(real_deltas)-61-58)[1:][::2], real_h_of_t_prime[real_hs][60:-60][1:][::2], marker='o', color='blue')
plt.plot(np.arange(59-58,len(real_deltas)-61-58)[::2], h_of_t_prime[hs][22:-23][::2], marker='o', color='tab:orange')
plt.plot(np.arange(59-58,len(real_deltas)-61-58)[1:][::2], h_of_t_prime[hs][22:-23][1:][::2], marker='o', color='orange')
plt.ylabel('Rise (Angstroms)')
plt.xlabel('Subunit index')
plt.show()

plt.tight_layout()
plt.rcParams.update({'font.size': 24})
plt.plot(np.arange(59-58,len(real_deltas)-61-58)[::2],real_deltas[60:-60][::2], marker='o', color='tab:blue')
plt.plot(np.arange(59-58,len(real_deltas)-61-58)[1:][::2],real_deltas[60:-60][1:][::2], marker='o', color='blue')
plt.plot(np.arange(59-58,len(real_deltas)-61-58)[::2],deltas[22:-23][::2], marker='o', color='tab:orange')
plt.plot(np.arange(59-58,len(real_deltas)-61-58)[1:][::2],deltas[22:-23][1:][::2], marker='o', color='orange')
plt.ylabel('Helical twist (degrees)')
plt.xlabel('Subunit index')
plt.show()


################################################################################
# Now, get curvature of the central axis
r_prime = final_axis_deriv.copy()
r_double_prime = final_axis_second_deriv.copy()
k = np.linalg.norm(np.cross(r_prime, r_double_prime), axis=-1) / (np.linalg.norm(r_prime, axis=-1)**3)


# The fun stuff: plot h(t), phi(t), K(t) as fxns of each other
curv = k[:-1][hs][1:-1]
rise = h_of_t_prime[hs][1:-1]
twist = np.degrees(phi_of_t_prime[hs][1:-1])

fig, ax = plt.subplots(1,3)
ax[0].scatter(rise, twist)
ax[1].scatter(1.0/curv, twist)
ax[2].scatter(1.0/curv,  rise)
plt.tight_layout()
plt.show()

plt.plot(1.0/k)
plt.ylim(-5000,10000)
plt.show()

################################################################################
# Color PDB by Curvature, Twist, Rise
p = parsePDB(mid_file_name)
rises = h_of_t_prime[hs][60:-60][::-1]
for i in range(0, len(rises)):
	p.select('chain '+string.ascii_uppercase[i+1]).setBetas(rises[i])

writePDB('14mer_riseBfactor.pdb', p)


p = parsePDB(mid_file_name)
phis = deltas[60:-60][::-1]
for i in range(0, len(phis)):
	p.select('chain '+string.ascii_uppercase[i+1]).setBetas(phis[i])

writePDB('14mer_phiBfactor.pdb', p)

p = parsePDB(mid_file_name)
ks = k[hs][60:-60][::-1]
for i in range(0, len(phis)):
	p.select('chain '+string.ascii_uppercase[i+1]).setBetas(ks[i]*10000)

writePDB('14mer_curvBfactor.pdb', p)











################################################################################
# Now verify
p_full = parsePDB(file_name)
subunit = p_full.select('chain A').copy()
subunit_coords = subunit.getCoords()
subunit_center = np.average(subunit.getCoords(), axis=0)
subunit_center = subunit_center / np.linalg.norm(subunit_center)
subunit_rot = np.eye(3)#get_R_from_a_to_b(subunit_center, np.array([1,0,0]))

subunit_standard = subunit.copy()
subunit_standard.setCoords(subunit_coords - np.average(subunit_coords, axis=0))
subunit_standard_coords = subunit_standard.getCoords()
rot_subunit_coords = []
for i in range(0, len(subunit_standard_coords)):
	rot_subunit_coords.append(rotate_pitch(np.radians(180),rotate_yaw(np.radians(0), rotate_roll(np.radians(180), np.matmul(subunit_rot, subunit_standard_coords[i])))))

rot_subunit_coords = np.asarray(rot_subunit_coords)

subunit_standard.setCoords(rot_subunit_coords)

showProtein(subunit_standard)
plt.show()



pts_on_final_axis = final_axis[hs]
normals_at_pts_on_final_axis = []
for i in range(0, len(rots[hs])):
	normals_at_pts_on_final_axis.append(np.matmul(rots[hs][i], rotate_yaw(phis[i]+np.radians(180),np.array([15.76719916835962,0,0]))))

normals_at_pts_on_final_axis = np.asarray(normals_at_pts_on_final_axis)
measured_centroids = pts_on_final_axis + normals_at_pts_on_final_axis

# plot the centroids in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
_=ax.scatter(measured_centroids[:,0], measured_centroids[:,1], measured_centroids[:,2], c='green', alpha=0.2, s=300)
_=ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], c='purple', alpha=0.4, s=300)
_=ax.scatter(avgs[:,0], avgs[:,1], avgs[:,2], c='red')
_=ax.set_xlim(-200,200); _=ax.set_ylim(-200,200); _=ax.set_zlim(-200,200)
plt.show()


# Still need to position the subunits correctly TODO
new_p = 0
for i in range(0, len(measured_centroids)):
	temp = subunit_standard.copy()
	rotated_temp_coords = []
	for j in range(0, len(temp.getCoords())):
		rotated_temp_coords.append(np.matmul(rots[hs][i], rotate_yaw(phis[i]+np.radians(180), np.array([15.76719916835962,0,0])+temp.getCoords()[j])))
	rotated_temp_coords = np.asarray(rotated_temp_coords)
	temp.setCoords(pts_on_final_axis[i]+rotated_temp_coords)
	temp.setChids(string.ascii_uppercase[i])
	if(new_p == 0):
		new_p = temp
	else:
		new_p = new_p + temp

writePDB('temp.pdb', new_p)

centroids_pf1 = centroids[::2]
centroids_pf2 = centroids[1::2]

centroids_pf1_dists = centroids[::2]
np.linalg.norm(centroids[::2][:-1] - centroids[::2][1:])






