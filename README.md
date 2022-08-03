# Bending Actin Project
This repository stores the files for "Bending forces and nucleotide state jointly regulate F-actin structure", currently available as a preprint at:
https://www.biorxiv.org/content/10.1101/2022.06.02.494606v1. This repository is currently organized into two sections:
1) particle\_picking 
2) analysis

## 1) particle_picking
This directory contains the scripts used to generate synthetic data, train neural networks, and pick on cryo-EM data

## 2) analysis
This directory contains the custom scripts used to analyze the data generated in this project.

## A note on running scripts
To ensure all necessary python packages are available when developing these scripts, an anaconda environement called matt_EMAN2 was used. If you would like to use the scripts provided in this repository, it is recommended that you make an anaconda environment using the provided yml file. You will also need to update the top line of the script to your installation of the anaconda environment. You may also use your own anaconda environments provided it has the necessary packages installed.
