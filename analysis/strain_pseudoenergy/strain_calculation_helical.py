#!/mnt/data1/Matt/anaconda_install/anaconda2/envs/matt_EMAN2/bin/python
################################################################################
# Import python files
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from prody import *
import string
################################################################################
straight_file_name = './ADP_goodH20_replacedBackbone-coot-6_real_space_refined_034_HICtoHIS.pdb'
bent_file_name = './ADP_Pi_alignedChainAtoADP_HICtoHIS.pdb'

output_file_name = 'shearStrain_helical.pdb'

################################################################################
def load_pdb(file_name):
	# Use ProDy to import PDB file
	p = parsePDB(file_name, subset='calpha')
	chids = set(p.getChids())
	chains = []
	for chain_idx in chids:
		chains.append(p.select('chain ' + chain_idx).copy())
	
	# get the coordinates for each atom of each actin subunit
	coords = []
	for i in range(0,len(chains)):
		coords.append(chains[i].getCoords())
	
	# make each helix into a [num_chains x num_atoms_per_actin x 3] array
	coords = np.asarray(coords)
	return coords

################################################################################
# Given two pdbs of the same dimensions, calculate the strain tensor (Em) and
# the shear energy (Es) per position along the alpha carbon chain
def calc_Em_Es(pdb0,pdb1):
	pdb0_distMatrix = buildDistMatrix(pdb0,pdb0)
	pdb1_distMatrix = buildDistMatrix(pdb1,pdb1)
	
	Es = np.zeros((len(pdb0))) # per-position shear energy
	Emout = np.zeros((len(pdb0),3,3)) # strain tensor
	radius = np.zeros((len(pdb0))) 
	for i in range(0, len(pdb0)):
		itnum = -1
		errnum = 0
		# need to deal with cases where the neighborhood has to be expanded to not 
		#permit strain energies to blow up
		while(errnum != itnum):
			weight_n = np.zeros((len(pdb0),1)) # initialize weight matrix to zeros
			weight_n[pdb0_distMatrix[:,i]<6] = 1 # set those atoms close to index to 1
			# Now get those alpha carbons that are more than 6 but less than 8 A away
			nlist = np.intersect1d(np.argwhere(pdb0_distMatrix[:,i]>=6), np.argwhere(pdb0_distMatrix[:,i]<=(8+errnum)))
			for j in range(0,len(nlist)):
				weight_n[nlist[j]] = 1-(0.5*(pdb0_distMatrix[nlist[j],i]-6))
			
			neighborhood = np.nonzero(weight_n)[0]
			wlist = weight_n[neighborhood]
			Dm = np.zeros((3,3))
			Am = np.zeros((3,3))
			for k in range(0, len(neighborhood)):
				dist0 = pdb0[neighborhood[k]] - pdb0[i]
				dist1 = pdb1[neighborhood[k]] - pdb1[i]
				Dm = Dm + (np.outer(dist0,dist0.T)*wlist[k])
				Am = Am + (np.outer(dist1,dist0.T)*wlist[k])
			
			Fm = np.matmul(Am,np.linalg.inv(Dm))
			Em = 0.5*(np.eye(3) - np.linalg.inv(np.matmul(Fm,Fm.T)))
			gamma = Em-((np.trace(Em)/3.0)*np.eye(3))
			Es[i]=np.sum(np.square(gamma))
			if (Es[i]>10):
				errnum=errnum+1
			
			Emout[i]=Em
			itnum=itnum+1
			radius[i]=8+errnum
	
	return Emout, Es

################################################################################
################################################################################
straight = load_pdb(straight_file_name)
bent = load_pdb(bent_file_name)

straight = straight.reshape(1113,3)
bent = bent.reshape(1113,3)

strain_tensors, shear_energies = [],[]
Em, Es = calc_Em_Es(straight, bent)
strain_tensors.append(Em)
shear_energies.append(Es)

strain_tensors = np.asarray(strain_tensors)
shear_energies = np.asarray(shear_energies)
shear_energies = shear_energies.reshape(3,371)

for i in range(0, len(shear_energies)):
	_=plt.plot(shear_energies[i])

plt.show()

################################################################################
# Now, color by shear strain
'''
p = parsePDB(straight_file_name, subset='calpha') #protein to map these shear energies to
chids = set(p.getChids())
chains = []
for chain_idx in chids:
	chains.append(p.select('chain ' + chain_idx).copy())

for i in range(0, len(shear_energies)):
	p.select('chain '+string.ascii_uppercase[i]).setBetas(np.log10(shear_energies[i]))

writePDB('shearStrain_curvStr_maptoStr.pdb', p)
'''
p = parsePDB(bent_file_name, subset='calpha') #protein to map these shear energies to
chids = set(p.getChids())
chains = []
for chain_idx in chids:
	chains.append(p.select('chain ' + chain_idx).copy())

for i in range(0, len(shear_energies)):
	p.select('chain '+string.ascii_uppercase[i]).setBetas(np.log10(shear_energies[i]))

writePDB(output_file_name, p)

################################################################################






