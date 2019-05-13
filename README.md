# Bending Actin Project
This repository stores the files for Matt's bent actin project. It is currently
organized into four different sections:
1) make\_bent\_actin: creates 
2) make\_synthetic\_data
3) train\_neural\_network
4) neural\_network\_predictions

## 1) make_bent_actin
In make\_bent\_actin, there are two provided scripts:
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

python 2.7;
numpy;
matplotlib;
tqdm;
scipy;
ProDy;

I have found it convenient to create an anaconda virtual environment with python version 2.7 and 
use pip to install these libraries (scipy and matplotlib should come with numpy). 
There are certainly other ways to install these libraries, but this way worked for me.

**Known Bugs/Issues:**

Currently none with this code. However, if the numSubunits is greater than 33, 
Chimera will not render the PDB properly, probably because it is too large. All 
of the information is still properly in the PDB file, it is a rendering issue 
with Chimera. PDBs with very large subunit numbers may be viewed in pymol.
<p>
	<img  src="https://github.com/alushinlab/bent_actin/tree/master/readme_imgs/many_actins_black.png" width="200"/>
</p>

![ScreenShot](https://raw.github.com/alushinlab/bent_actin/tree/master/readme_imgs/many_actins_black.png)

## 2) make_synthetic_data
In make\_synthetic\_data there is one executable python file called 'projection\_generator.py'
This script requires as input a directory containing an arbitrary number of .mrc files.
It currently rotates the MRC file in plane and translates in x,y,z using EMAN2 
transformation functions. Then it convolves the MRC volume with a CTF using a SPARX 
function, with a randomly chosen defocus value. Then the volume is projected in Z
and noise is added in Fourier space to the 2D projection. 

I generated my .mrc files by creating several .pdb files, as described above, and 
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

**Required to run:**

python 2.7;
numpy;
tqdm;
scipy;

sparx;
EMAN2;
mrcfile;

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

## 3) train_neural_network
This script trains a contractive, denoising autoencoder to reconstruct a noiseless
image from a noisy input. At a future date, I will include the neural network's 
architecture, but for now I will say that I have seven outer convolutional layers
followed by three inner dense/fully-connected layers to encode a noisy image 
to a 128-dimensional vector, and the mirror of those layers is used to decode that
vector. 
The network is trained by first greedily training the convolutional layers until 
just before overfitting, then I train the inner layers for one epoch. Lastly, I 
train the full network on a large dataset.

**The script for this is still under development. I will update the README.md documentation when it is close to being in its final form**

**Required to run:**

python 2.7;
numpy;
matplotlib;
tqdm;
scipy;

tensorflow-gpu;
keras (using tensorflow as backend);

mrcfile;

## 4) neural_network_predictions
Currently this code is at the end of the previous script. I will decouple these 
two scripts in my next iteration so that I can first train a network, and then 
once that network is trained, I can use it to make predictions and see my encoded 
representations of input images. 





