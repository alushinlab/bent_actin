import chimera
from chimera import runCommand as rc
import os
################################################################################
#os.chdir('pdbs/PC1')

def stitch_actin_filaments(model_number):
	for i in range(0, 3):
		rc('open ../isolde_results/ADP_Pi_cryodrgn_isolde_frame%s.pdb'%model_number)
		#rc('open ../isolde_results/ADP_Pi_cryodrgn_isolde_frame%s.pdb'%str(model_number).zfill(3))

	rc('matchmaker #0:.A:.B:.C #1:.N:.O:.P pairing ss')
	#rc('delete #1:.W:.X:.Y')
	rc('delete #0:.A:.B:.C')
	rc('matchmaker #1:.A:.B:.C #2:.N:.O:.P pairing ss')
	rc('delete #2:.N:.O:.P')

	#rc('matchmaker #0:.X:.Y:.Z #3:.A:.B:.C pairing ss')
	#rc('delete #3:.A:.B:.C')
	#rc('matchmaker #3:.X:.Y:.Z #4:.A:.B:.C pairing ss')
	#rc('delete #4:.A:.B:.C')

	rc('combine #0,1,2 close true')
	rc('write #3 ADP_Pi_cryodrgn_isolde_frame%s_extended.pdb'%model_number)
	#rc('write #0 ADP_cryodrgn_isolde_frame%s_A.pdb'%model_number)
	#rc('write #1 ADP_cryodrgn_isolde_frame%s_B.pdb'%model_number)
	#rc('write #2 ADP_cryodrgn_isolde_frame%s_C.pdb'%model_number)
	rc('close #3')

for i in range(0, 10):
	stitch_actin_filaments(str(i).zfill(3))

rc('stop now')





