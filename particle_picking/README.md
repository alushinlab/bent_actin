# Neural network-based particle picking
This repository stores the files used for particle picking using a custom neural network approach. It is currently
organized into four different sections:
1) make\_bent\_actin: creates 
2) generate\_synthetic\_data
3) train\_neural\_networks
4) particle\_picking\_scripts

## 1) make_bent_actin
In make\_bent\_actin, there are two provided python files:
- generate\_bent\_actin\_streamlined.py - generates the actin filament
- generate\_bent\_actin\_helper\_fxns.py - contains helper functions called
by generate\_bent\_actin\_streamlined.py



Have both python files in your current directory, make generate\_bent\_actin\_streamlined.py 
executable, have 'actin\_28mer\_centered.pdb' in your current directory, and 
enter the following command:
```
./generate_bent_actin_streamlined.py --subunits numSubunits --radius radCurv --outputFileName out.pdb --plotting True
```
Where:
- numSubunits = the number of total subunits for this bent actin filament. MUST BE ODD
- radCurv = the radius of curvature for this bent actin. Typically a float between 150.0 to 100000.0; radCurv may be negative
- outputFileName = Output file name. If not specified, will default to bent_actin_NN_subunits_RRR_curvature.pdb
- plotting = Set to False if you do not want to see centroids plot. Leave blank or True otherwise
 
**Required to run:**

A PDB file of actin in your current directory. Currently, it must be called 'actin\_28mer\_centered.pdb' 
but this can be changed in the future if we want to use different models.

**Known Bugs/Issues:**

Currently none with this code. However, if the numSubunits is greater than 33, 
Chimera will not render the PDB properly, probably because it is too large. All 
of the information is still properly in the PDB file, it is a rendering issue 
with Chimera. PDBs with very large subunit numbers may be viewed in pymol.

**Example results:**
The synthetic volumes/models used for this project can be found at https://doi.org/10.5281/zenodo.6917913 in the synthetic_volumes_and_pdb.zip file.

## 2) generate_synthetic_data
In generate\_synthetic\_data there are two executable python files.
These scripts requires as input a directory containing an arbitrary number of .mrc files.
The program rotates the MRC file using euler angles and translates in x,y,z using EMAN2 
transformation functions. Then it convolves the MRC volume with a CTF using a SPARX 
function, with a randomly chosen defocus value. Then the volume is projected in Z
and noise is added in Fourier space to the 2D projection. 

I generated my .mrc files by creating several .pdb files, as described using the generate_bent_actin_streamlined.py script, and 
converting them to .mrc format using EMAN2's pdb2mrc function. You may also use 
chimera to make these maps if you so desire:
```
pdb2mrc input.pdb output.mrc apix=1.03 res=2.5 box=512 
```
Where:
- input.pdb = path to input PDB file to be turned into an MRC file
- output.mrc = output file name to save MRC file to
- apix = pixel size in angstroms per voxel
- res = resolution to which the .mrc file will be lowpass filtered
- box = box size in voxels

**To Run projection_generator.py**

Have a directory holding your MRC files that will be projected. There may be 
other files in this directory, but all files ending in .mrc must be 3D volumes

Have two output directories to store the noisy and noiseless projections. Ideally
these files will be empty, but if not, they must at least not contain any .json files.
```
./projection_generator.py --input_mrc_dir input_dir --output_noise_dir output_noise --output_noiseless_dir output_noNoise --numProjs projectionNumber --nProcs threadNumber
```
**Outputs**
- Noisy 2D projections
- Noiseless 2D projections
- json files storing parameters corresponding to that projection (you can take 
the specified actin file and apply these parameters to generate that projection).

**Some Notes**

I merge the json files at the end into a file called master.json.

numProjs should be a multiple of nProcs. If it is not, the program will round 
numProjs down to be a multiple of nProcs.


Because I am currently only trying to find large conformational changes, I have 
found it useful to lowpass filter my noiseless 2D projections to 15 angstroms.
To do so, I copy my noiseless projections to a new directory and run:
```
for file in ./*
do
proc3d ${file} ${file} lp=15 apix=1.03 norm
done
```
This command will lowpass filter each projection to 15 angstroms and normalize it.

**Known Bugs/Issues:**

- In the json file output, the 'actin_num' key refers to the position in the
actin model array that the MRC file was loaded into. I should update this to instead 
be the file name, not the position in the array.
- Sometimes, after the script has executed, python will print messages about the
job finishing. I am not sure why these outputs show up or how to suppress them.
They do not effect functionality. 
- If you try to kill the script while it is running, it might not be terminated
properly. This is a consequence of how python handles multiprocessing. I would 
recommend not killing the script once started. If it must be terminated, closing
the terminal should stop it.

**Example results:**
The training data for the DAE used in this project can be found at https://doi.org/10.5281/zenodo.6917913 in the DAE_synthetic_data.zip file.
The training data for the FCN-SS used in this project can be found at https://doi.org/10.5281/zenodo.6917913 in the semanticSegmentation_synthetic_data.zip file.




## 3) train_neural_networks
This directory contains the scripts used to train neural networks that may then be used for particle picking. CDAE.py trains a contractive, denoising autoencoder that is able to reconstruct a denoised image from a noisy input. The network architecture employed in this project has been largely replaced by a fully convolutional approach (https://github.com/alushinlab/plastin_bundles), but the dense layers used in this version have interesting implications in terms of manifold learning in the context of filament bending.

**To Run CDAE.py**

Have two directories containing noisy/noiseless projection pairs.

Hard code the number of images from the training set that you would like to load as well as the output file names. You may also change hard-coded parameters specific to training to optimize it for a particular use case. The program does not accept command-line arguments, so to run it simply enter:
```
./CDAE.py
```

**Outputs**
- Saved neural network in .h5 file

**To Run train_FCN_for_semseg.py**

Provide similar input as for CDAE.py, but provide the directory to a set of noisy projections and the corresponding set of semantically segmented images.

Hard code the number of images from the training set that you would like to load as well as the output file names. You may also change hard-coded parameters specific to training to optimize it for a particular use case. The program does not accept command-line arguments, so to run it simply enter:
```
./train_FCN_for_semseg.py
```
**Outputs**
- Saved neural network in .h5 file

**Example results:**
The trained neural networks used for this project can be found at https://doi.org/10.5281/zenodo.6917913 in the trained_neural_networks.zip file.




## 4) particle_picking_scripts
A set of particle picking scripts are provided. I will outlin an example use case of one of them:
 - Set up a RELION directory and process the data through motion correction and CTF estimation
 - In the RELION directory, make a directory called Micrographs_bin4 and bin the motion-corrected micrographs by 4
 - In the RELION directory, make a directory called particle_picking
 - In particle_picking, make three directories: trained_networks, pngs, starFiles
 - Copy trained neural networks to trained_networks


**To Run the Picking Script**

Hard code specific picking parameters, including the GPU identity you want to use. The program does not accept command-line arguments, so to run it simply enter:
```
./multiple_whole_micrograph_preds.py
```
If you are using a workstation with multiple GPUs, you may run multiple instances of the script in separate terminals. This speeds up picking linearly.





