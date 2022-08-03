#!/mnt/data1/Matt/anaconda_install/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
# Import python files
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from prody import *
import string; import sys
from measure_tilt_helper import *
import sys
################################################################################
################################################################################
#file_name = '../isolde_results/ADP_cryodrgn_isolde_frame%s.pdb'%str(sys.argv[1]).zfill(3)
file_name = '../isolde_results/ADP_cryodrgn_isolde_frame%s.pdb'%str(9).zfill(3)
#file_name = '../isolde_results/ADP_Pi_cryodrgn_isolde_frame%s.pdb'%str(9).zfill(3)
#file_name = '../isolde_results/final_ADP_Pi_AtoPhelix.pdb'
coords, disp_from_orig = load_coords(file_name)
#coords = coords - coords.reshape(coords.shape[0]*coords.shape[1],3).mean(axis=0)
subdomain_coords = convert_coords_to_subdomains(coords)

################################################################################
# Intrasubunit measures
# B0_1 = bond length between SD1 and SD2
B0_1 = np.zeros((16))
for i in range(0, len(subdomain_coords)):
	B0_1[i] = np.linalg.norm(subdomain_coords[i][0] - subdomain_coords[i][1])

# B0_2 = bond length between SD1 and SD3
B0_2 = np.zeros((16))
for i in range(0, len(subdomain_coords)):
	B0_2[i] = np.linalg.norm(subdomain_coords[i][0] - subdomain_coords[i][2])

# B0_3 = bond length between SD3 and SD4
B0_3 = np.zeros((16))
for i in range(0, len(subdomain_coords)):
	B0_3[i] = np.linalg.norm(subdomain_coords[i][2] - subdomain_coords[i][3])

# B0_4 = bond length between SD1 and SD4
B0_4 = np.zeros((16))
for i in range(0, len(subdomain_coords)):
	B0_4[i] = np.linalg.norm(subdomain_coords[i][0] - subdomain_coords[i][3])

# B0_5 = bond length between SD2 and SD3
B0_5 = np.zeros((16))
for i in range(0, len(subdomain_coords)):
	B0_5[i] = np.linalg.norm(subdomain_coords[i][1] - subdomain_coords[i][2])

# B0_6 = bond length between SD2 and SD4
B0_6 = np.zeros((16))
for i in range(0, len(subdomain_coords)):
	B0_6[i] = np.linalg.norm(subdomain_coords[i][1] - subdomain_coords[i][3])



# A0_1 = angle between 2-1-3
A0_1 = np.zeros((16))
for i in range(0, len(subdomain_coords)):
	A0_1[i] = three_pt_angle(subdomain_coords[i][1], subdomain_coords[i][0], subdomain_coords[i][2])

# A0_2 = angle between 1-3-4
A0_2 = np.zeros((16))
for i in range(0, len(subdomain_coords)):
	A0_2[i] = three_pt_angle(subdomain_coords[i][0], subdomain_coords[i][2], subdomain_coords[i][3])

# A0_3 = angle between 3-4-2
A0_3 = np.zeros((16))
for i in range(0, len(subdomain_coords)):
	A0_3[i] = three_pt_angle(subdomain_coords[i][2], subdomain_coords[i][3], subdomain_coords[i][1])

# A0_4 = angle between 4-2-1
A0_4 = np.zeros((16))
for i in range(0, len(subdomain_coords)):
	A0_4[i] = three_pt_angle(subdomain_coords[i][3], subdomain_coords[i][1], subdomain_coords[i][0])

# A0_5 = angle between 1-2-3
A0_5 = np.zeros((16))
for i in range(0, len(subdomain_coords)):
	A0_5[i] = three_pt_angle(subdomain_coords[i][0], subdomain_coords[i][1], subdomain_coords[i][2])

# A0_6 = angle between 2-3-1
A0_6 = np.zeros((16))
for i in range(0, len(subdomain_coords)):
	A0_6[i] = three_pt_angle(subdomain_coords[i][1], subdomain_coords[i][2], subdomain_coords[i][0])

# A0_7 = angle between 2-1-4
A0_7 = np.zeros((16))
for i in range(0, len(subdomain_coords)):
	A0_7[i] = three_pt_angle(subdomain_coords[i][1], subdomain_coords[i][0], subdomain_coords[i][3])

# A0_8 = angle between 1-4-2
A0_8 = np.zeros((16))
for i in range(0, len(subdomain_coords)):
	A0_8[i] = three_pt_angle(subdomain_coords[i][0], subdomain_coords[i][3], subdomain_coords[i][1])

# A0_9 = angle between 2-3-4
A0_9 = np.zeros((16))
for i in range(0, len(subdomain_coords)):
	A0_9[i] = three_pt_angle(subdomain_coords[i][1], subdomain_coords[i][2], subdomain_coords[i][3])

# A0_10 = angle between 4-2-3
A0_10 = np.zeros((16))
for i in range(0, len(subdomain_coords)):
	A0_10[i] = three_pt_angle(subdomain_coords[i][3], subdomain_coords[i][1], subdomain_coords[i][2])


# D0_1 = dihedral angle between 2-1-3-4
D0_1 = np.zeros((16))
for i in range(0, len(subdomain_coords)):
	D0_1[i] = compute_dihedral(subdomain_coords[i][[1,0,2,3]])

B0_1.mean(),B0_2.mean(),B0_3.mean(),A0_1.mean(),A0_2.mean(),D0_1.mean()

intrasubunit = np.asarray([B0_1,B0_2,B0_3,B0_4,B0_5,B0_6,A0_1,A0_2,A0_3,A0_4,
A0_5,A0_6,A0_7,A0_8,A0_9,D0_1,A0_10])

fig, ax = plt.subplots(4,4)
for i in range(0,4):
	for j in range(0,4):
		ax[i,j].plot(range(0,len(intrasubunit[0]))[::2], intrasubunit[4*i+j][::2])
		ax[i,j].plot(range(0,len(intrasubunit[0]))[1::2], intrasubunit[4*i+j][1::2])

ax[0,0].set_ylabel('SD1-SD2 distance')
ax[0,1].set_ylabel('SD1-SD3 distance')
ax[0,2].set_ylabel('SD3-SD4 distance')
ax[1,0].set_ylabel('SD2-SD1-SD3 angle')
ax[1,1].set_ylabel('SD1-SD3-SD4 angle')
ax[1,2].set_ylabel('SD2-SD1-SD3-SD4 dihedral angle')
fig.set_size_inches(16, 9)
plt.savefig('tilt_angles/all_measures/intrasubunit_' + file_name.split('/')[-1].split('.')[0])
plt.cla()
#plt.show()


################################################################################
# Minor sites

# Interstrand
# B1_1 = bond length between SD2 and SD3 on the next strand
B1_1 = np.zeros((15))
for i in range(0, len(subdomain_coords)-1):
	B1_1[i] = np.linalg.norm(subdomain_coords[i][1] - subdomain_coords[i+1][2])

# B1_2 = bond length between SD4 and SD1 on the next strand
B1_2 = np.zeros((15))
for i in range(0, len(subdomain_coords)-1):
	B1_2[i] = np.linalg.norm(subdomain_coords[i][3] - subdomain_coords[i+1][0])

# B1_3 = bond length between SD4 and SD3 on the next strand
B1_3 = np.zeros((15))
for i in range(0, len(subdomain_coords)-1):
	B1_3[i] = np.linalg.norm(subdomain_coords[i][3] - subdomain_coords[i+1][2])

# B1_4 = bond length between SD2 and SD4 on the next strand
B1_4 = np.zeros((15))
for i in range(0, len(subdomain_coords)-1):
	B1_4[i] = np.linalg.norm(subdomain_coords[i][1] - subdomain_coords[i+1][3])

# B1_5 = bond length between SD1 and SD4 on the next strand
B1_5 = np.zeros((15))
for i in range(0, len(subdomain_coords)-1):
	B1_5[i] = np.linalg.norm(subdomain_coords[i][0] - subdomain_coords[i+1][3])

# B1_6 = bond length between SD1 and SD1 on the next strand
B1_6 = np.zeros((15))
for i in range(0, len(subdomain_coords)-1):
	B1_6[i] = np.linalg.norm(subdomain_coords[i][0] - subdomain_coords[i+1][0])

# B1_7 = bond length between SD2 and SD2 on the next strand
B1_7 = np.zeros((15))
for i in range(0, len(subdomain_coords)-1):
	B1_7[i] = np.linalg.norm(subdomain_coords[i][1] - subdomain_coords[i+1][1])

# B1_8 = bond length between SD3 and SD3 on the next strand
B1_8 = np.zeros((15))
for i in range(0, len(subdomain_coords)-1):
	B1_8[i] = np.linalg.norm(subdomain_coords[i][2] - subdomain_coords[i+1][2])

# B1_9 = bond length between SD4 and SD4 on the next strand
B1_9 = np.zeros((15))
for i in range(0, len(subdomain_coords)-1):
	B1_9[i] = np.linalg.norm(subdomain_coords[i][3] - subdomain_coords[i+1][3])



# A1_1 = angle between 3-4-3_1
A1_1 = np.zeros((15))
for i in range(0, len(subdomain_coords)-1):
	A1_1[i] = three_pt_angle(subdomain_coords[i][2], subdomain_coords[i][3], subdomain_coords[i+1][2])

# A1_2 = angle between 4-3_1-4_1
A1_2 = np.zeros((15))
for i in range(0, len(subdomain_coords)-1):
	A1_2[i] = three_pt_angle(subdomain_coords[i][3], subdomain_coords[i+1][2], subdomain_coords[i+1][3])

# D1_1 = dihedral angle between 3-4-3_1-4_1
D1_1 = np.zeros((15))
for i in range(0, len(subdomain_coords)-1):
	D1_1[i] = compute_dihedral([subdomain_coords[i][2],subdomain_coords[i][3],subdomain_coords[i+1][2],subdomain_coords[i+1][3]])

# D1_2 = dihedral angle between 1-2-1_1-2_1
D1_2 = np.zeros((15))
for i in range(0, len(subdomain_coords)-1):
	D1_2[i] = compute_dihedral([subdomain_coords[i][0],subdomain_coords[i][1],subdomain_coords[i+1][0],subdomain_coords[i+1][1]])


interstrand = np.asarray([B1_1,B1_2,B1_3,B1_4,B1_5,B1_6,B1_7,B1_8,B1_9,A1_1,A1_2,D1_1,D1_2])
fig, ax = plt.subplots(4,4)
for i in range(0,4):
	for j in range(0,4):
		ax[i,j].plot(range(0,len(interstrand[0]))[::2], interstrand[4*i+j][::2])
		ax[i,j].plot(range(0,len(interstrand[0]))[1::2], interstrand[4*i+j][1::2])

plt.show()


i=11
plt.plot(range(0,len(interstrand[0]))[::2], interstrand[i][::2])
plt.plot(range(0,len(interstrand[0]))[1::2], interstrand[i][1::2])
plt.xlabel('Subunit Index')
plt.ylabel('Dihedral angle between SD3-SD4-SD3_+1-SD4_+1')
plt.show()

################################################################################
# Intrastrand
# B2_1 = bond length between SD2 and SD1 on the next subunit same strand
B2_1 = np.zeros((14))
for i in range(0, len(subdomain_coords)-2):
	B2_1[i] = np.linalg.norm(subdomain_coords[i][1] - subdomain_coords[i+2][0])

# B2_2 = bond length between SD2 and SD3 on the next subunit same strand
B2_2 = np.zeros((14))
for i in range(0, len(subdomain_coords)-2):
	B2_2[i] = np.linalg.norm(subdomain_coords[i][1] - subdomain_coords[i+2][2])

# B2_3 = bond length between SD4 and SD3 on the next subunit same strand
B2_3 = np.zeros((14))
for i in range(0, len(subdomain_coords)-2):
	B2_3[i] = np.linalg.norm(subdomain_coords[i][3] - subdomain_coords[i+2][2])

# B2_4 = bond length between SD1 and SD1 on the next subunit same strand
B2_4 = np.zeros((14))
for i in range(0, len(subdomain_coords)-2):
	B2_4[i] = np.linalg.norm(subdomain_coords[i][0] - subdomain_coords[i+2][0])

# B2_5 = bond length between SD4 and SD4 on the next subunit same strand
B2_5 = np.zeros((14))
for i in range(0, len(subdomain_coords)-2):
	B2_5[i] = np.linalg.norm(subdomain_coords[i][3] - subdomain_coords[i+2][3])

# B2_6 = bond length between SD4 and SD1 on the next subunit same strand
B2_6 = np.zeros((14))
for i in range(0, len(subdomain_coords)-2):
	B2_6[i] = np.linalg.norm(subdomain_coords[i][3] - subdomain_coords[i+2][0])

# A2_1 = angle between 1-2-3_2
#A2_1 = np.zeros((14))
#for i in range(0, len(subdomain_coords)-2):
#	A2_1[i] = three_pt_angle(subdomain_coords[i][0], subdomain_coords[i][1], subdomain_coords[i+2][2])
# A2_1 = angle between 1-4_1-1_2
A2_1 = np.zeros((14))
for i in range(0, len(subdomain_coords)-2):
	A2_1[i] = three_pt_angle(subdomain_coords[i][0], subdomain_coords[i+1][1], subdomain_coords[i+2][0])




# A2_2 = angle between 2-3_2-4_2
A2_2 = np.zeros((14))
for i in range(0, len(subdomain_coords)-2):
	A2_2[i] = three_pt_angle(subdomain_coords[i][1], subdomain_coords[i+2][2], subdomain_coords[i+2][3])

# A2_3 = angle between 1-4-1_2
A2_3 = np.zeros((14))
for i in range(0, len(subdomain_coords)-2):
	A2_3[i] = three_pt_angle(subdomain_coords[i][0], subdomain_coords[i][3], subdomain_coords[i+2][0])

# A2_4 = angle between 1-2-1_2
A2_4 = np.zeros((14))
for i in range(0, len(subdomain_coords)-2):
	A2_4[i] = three_pt_angle(subdomain_coords[i][0], subdomain_coords[i][1], subdomain_coords[i+2][0])

# A2_5 = angle between 4-3_2-4_2
A2_5 = np.zeros((14))
for i in range(0, len(subdomain_coords)-2):
	A2_5[i] = three_pt_angle(subdomain_coords[i][3], subdomain_coords[i+2][2], subdomain_coords[i+2][3])

# A2_6 = angle between 4-1_2-4_2
A2_6 = np.zeros((14))
for i in range(0, len(subdomain_coords)-2):
	A2_6[i] = three_pt_angle(subdomain_coords[i][3], subdomain_coords[i+2][0], subdomain_coords[i+2][3])

# A2_7 = angle between 4-1_2-4_2
A2_7 = np.zeros((14))
for i in range(0, len(subdomain_coords)-2):
	A2_7[i] = three_pt_angle(subdomain_coords[i][3], subdomain_coords[i+2][0], subdomain_coords[i+2][3])

# D2_1 = dihedral angle between 1-2-3_2-4_2
D2_1 = np.zeros((14))
for i in range(0, len(subdomain_coords)-2):
	D2_1[i] = compute_dihedral([subdomain_coords[i][0],subdomain_coords[i][1],subdomain_coords[i+2][2],subdomain_coords[i+2][3]])

for i in range(0, len(D2_1)):
	if(D2_1[i] <0):
		D2_1[i] = D2_1[i]+360

# D2_2 = dihedral angle between 1-4-1_2-4_2
D2_2 = np.zeros((14))
for i in range(0, len(subdomain_coords)-2):
	D2_2[i] = compute_dihedral([subdomain_coords[i][0],subdomain_coords[i][3],subdomain_coords[i+2][0],subdomain_coords[i+2][3]])


intrastrand = np.asarray([B2_1,B2_2,B2_3,B2_4,B2_5,B2_6,A2_1,A2_2,A2_3,A2_4,A2_5,A2_6,D2_1,D2_2])
fig, ax = plt.subplots(4,4)
for i in range(0,4):
	for j in range(0,4):
		ax[i,j].plot(range(0,len(intrastrand[0]))[::2], intrastrand[4*i+j][::2])
		ax[i,j].plot(range(0,len(intrastrand[0]))[1::2], intrastrand[4*i+j][1::2])

plt.show()


fig, ax = plt.subplots(3)
ax[0].plot(range(0,len(intrastrand[0]))[::2], intrastrand[0][::2])
ax[0].plot(range(0,len(intrastrand[0]))[1::2], intrastrand[0][1::2])
ax[1].plot(range(0,len(intrastrand[0]))[::2], intrastrand[3][::2])
ax[1].plot(range(0,len(intrastrand[0]))[1::2], intrastrand[3][1::2])
ax[2].plot(range(0,len(intrastrand[0]))[::2], intrastrand[4][::2])
ax[2].plot(range(0,len(intrastrand[0]))[1::2], intrastrand[4][1::2])
ax[2].set_xlabel('Subunit index')
plt.show()


fig, ax = plt.subplots(2)
ax[0].plot(range(0,len(intrastrand[0]))[::2], intrastrand[8][::2])
ax[0].plot(range(0,len(intrastrand[0]))[1::2], intrastrand[8][1::2])
ax[1].plot(range(0,len(intrastrand[0]))[::2], intrastrand[9][::2])
ax[1].plot(range(0,len(intrastrand[0]))[1::2], intrastrand[9][1::2])
ax[1].set_xlabel('Subunit index')
plt.show()


fig, ax = plt.subplots(2)
ax[0].plot(range(0,len(D1_1))[::2], D1_1[::2])
ax[0].plot(range(0,len(D1_1))[1::2], D1_1[1::2])
ax[1].plot(range(0,len(A2_1))[::2], A2_1[::2])
ax[1].plot(range(0,len(A2_1))[1::2], A2_1[1::2])
ax[1].set_xlabel('Subunit index')
plt.show()


################################################################################
# looks like as bending happens, SD1-SD1_2 distance increases (so does SD2-SD1_2
# distance). Also, 1-4-1_2 angle changes to match. How does this happen?
# I predict SD2 changes in volume - compute convex hull and get enclosed volume

# Trying to fit adp-pi into it, I need to check if SD1-4 distance changes differently
# for intrasubunit distance between Pi conditions
hull_vols = np.zeros((14))
SD2_coordinates = SD2_coords(coords)
from scipy.spatial import ConvexHull
for i in range(0, len(hull_vols)):
	hull = ConvexHull(SD2_coordinates[i])
	hull_vols[i] = hull.area


plt.plot(range(0,len(hull_vols))[::2], hull_vols[::2])
plt.plot(range(1,len(hull_vols))[::2], hull_vols[1::2])
plt.show()







sys.exit()

################################################################################
################################################################################
# winning measures
# B2_1 = bond length between SD2 and SD1 on the next subunit same strand
# B2_4 = bond length between SD1 and SD1 on the next subunit same strand
# B2_5 = bond length between SD4 and SD4 on the next subunit same strand
# A2_1 = angle between 1-4_1-1_2
# A2_2 = angle between 2-3_2-4_2
# A2_3 = angle between 1-4-1_2
# A2_4 = angle between 1-2-1_2
# A2_5 = angle between 4-3_2-4_2
# A2_7 = angle between 4-1_2-4_2
# D1_1 = dihedral angle between 3-4-3_1-4_1
# D2_1 = dihedral angle between 1-2-3_2-4_2
titles = np.asarray(['dist btw SD2 and SD1 \non the next subunit same strand',
'dist btw SD1 and SD1 \non the next subunit same strand',
'dist btw SD4 and SD4 \non the next subunit same strand',
'angle between 1-4_1-1_2',
'angle between 2-3_2-4_2',
'angle between 1-4-1_2',
'angle between 1-2-1_2',
'angle between 4-3_2-4_2',
'angle between 4-1_2-4_2',
'dihedral angle btwn 3-4-3_1-4_1',
'dihedral angle btwn 1-2-3_2-4_2'])

good_measures = np.asarray([B2_1,B2_4,B2_5,A2_1,A2_2,A2_3,A2_4,A2_5,A2_7,D1_1,D2_1])
fig, ax = plt.subplots(3,4)
for i in range(0,3):
	for j in range(0,4):
		ax[i,j].plot(range(0,len(good_measures[4*i+j]))[::2], good_measures[4*i+j][::2])
		ax[i,j].plot(range(0,len(good_measures[4*i+j]))[1::2], good_measures[4*i+j][1::2])
		ax[i,j].set_title(titles[4*i+j])

fig.set_size_inches(16, 9)
fig.tight_layout()
plt.show()

good_measures_forFig = [B2_1,B2_4,B2_5,A2_1,A2_3,A2_4,A2_5,D1_1,D0_1]
#np.save('measures_forFig_ADP_Pi_frame009.npy', good_measures_forFig)

import pandas as pd
df1 = pd.DataFrame(good_measures_forFig).T
df1.columns = ['B2_1 = dist btwn SD2 and SD1 on the next subunit same strand',
					'B2_4 = dist btwn SD1 and SD1 on the next subunit same strand',
					'B2_5 = dist btwn SD4 and SD4 on the next subunit same strand',
					'A2_1 = angle btwn 1-4_1-1_2',
					'A2_3 = angle btwn 1-4-1_2',
					'A2_4 = angle btwn 1-2-1_2',
					'A2_5 = angle btwn 4-3_2-4_2',
					'D1_1 = dihedral angle btwn 3-4-3_1-4_1',
					'D0_1 = dihedral angle btwn 2-1-3-4 on the same subunit'
					]
df1.to_csv('ADP_goodMeasures_forFig.csv')

################################################################################
# save coordinates as bild file
o='frame_009_2.bild'
out=open(o, 'w')
cntr_2=0
for i in range(0, len(subdomain_coords)):
	for j in range(0, len(subdomain_coords[i])):
		#write out marker entries for each residue pair
		out.write('.color %.4f %.4f %.4f\n'%((1.0/14.0)*i,0.5+0.05*j,0.5))
		out.write(".sphere %.5f %.5f %.5f %.5f \n"%(subdomain_coords[i][j][0], subdomain_coords[i][j][1], subdomain_coords[i][j][2], 6))
	
	out.write('.color %.4f %.4f %.4f\n'%(0.5,0.5,0.5))
	out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(subdomain_coords[i][1][0], subdomain_coords[i][1][1], subdomain_coords[i][1][2], subdomain_coords[i][0][0], subdomain_coords[i][0][1], subdomain_coords[i][0][2], 2))
	out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(subdomain_coords[i][0][0], subdomain_coords[i][0][1], subdomain_coords[i][0][2], subdomain_coords[i][2][0], subdomain_coords[i][2][1], subdomain_coords[i][2][2], 2))
	out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(subdomain_coords[i][2][0], subdomain_coords[i][2][1], subdomain_coords[i][2][2], subdomain_coords[i][3][0], subdomain_coords[i][3][1], subdomain_coords[i][3][2], 2))


cmap = plt.get_cmap('bwr')
B2_1_cmap_vals = ((B2_1 - B2_1.mean()) / np.max(np.abs(B2_1 - B2_1.mean())) +1 )/2.0
B2_1_cmap = cmap(B2_1_cmap_vals)

cmap = plt.get_cmap('bwr')
B2_5_cmap_vals = ((B2_5 - B2_5.mean()) / np.max(np.abs(B2_5 - B2_5.mean())) +1 )/2.0
B2_5_cmap = cmap(B2_5_cmap_vals)

for i in range(0, len(subdomain_coords)-2):
	#SD1-SD1_2 distance
	#out.write('.color %.4f %.4f %.4f\n'%(B2_1_cmap[i][0],B2_1_cmap[i][1],B2_1_cmap[i][2]))
	#out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(subdomain_coords[i][0][0], subdomain_coords[i][0][1], subdomain_coords[i][0][2], subdomain_coords[i+2][0][0], subdomain_coords[i+2][0][1], subdomain_coords[i+2][0][2], 3))
	#SD4-SD4_2 distance
	out.write('.color %.4f %.4f %.4f\n'%(B2_5_cmap[i][0],B2_5_cmap[i][1],B2_5_cmap[i][2]))
	out.write(".cylinder %.5f %.5f %.5f %.5f %.5f %.5f %.5f \n"%(subdomain_coords[i][3][0], subdomain_coords[i][3][1], subdomain_coords[i][3][2], subdomain_coords[i+2][3][0], subdomain_coords[i+2][3][1], subdomain_coords[i+2][3][2], 3))


#write final line of xml file, is constant	
out.close()	







good_measures_hold_ADP006 = good_measures[[0,1,5,8]]
good_measures_hold_ADP_Pi008 = good_measures[[0,1,5,8]]






fig, ax = plt.subplots(2,2)
for i in range(0,2):
	for j in range(0,2):
		ax[i,j].plot(range(0,len(good_measures_hold_ADP006[2*i+j]))[::2], good_measures_hold_ADP006[2*i+j][::2])
		ax[i,j].plot(range(0,len(good_measures_hold_ADP006[2*i+j]))[1::2], good_measures_hold_ADP006[2*i+j][1::2])
		ax[i,j].set_title(titles[4*i+j])

for i in range(0,2):
	for j in range(0,2):
		ax[i,j].plot(range(0,len(good_measures_hold_ADP_Pi008[2*i+j]))[::2], good_measures_hold_ADP_Pi008[2*i+j][::2],c='blue')
		ax[i,j].plot(range(0,len(good_measures_hold_ADP_Pi008[2*i+j]))[1::2], good_measures_hold_ADP_Pi008[2*i+j][1::2],c='orange')
		ax[i,j].set_title(titles[4*i+j])


fig.set_size_inches(6, 6)
fig.tight_layout()
plt.show()





