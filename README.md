# MRI-Pipelines
This repository contains the code which I use to analyze fMRI data. 


-------------------------------------------------------------------------------------------------------------
dti_preprocessing_ACI.py

This file contains code to run the most up to date gpu and non-gpu counter parts of FSL functions for the anaylsis of diffusion tensor imaging data (DTI). Currently, I specificy inside of the script where the files are on the computer, but his will be changed in the future to be more user friendly. Additionally, I have currently commented out the GPU function for eddy cuda since it is does not use the same cuda toolkit version as the bedpostx_gpu function that I am using. For users who wish to use both, you will need to make sure that you have the bedpostx_gpu version for cuda7.5, not cuda8.0. If you do not want to use the GPU functions, you can comment them out and uncomment out the non_gpu functions. All of the GPU functions are labeled with 'gpu' or 'cuda' in the name. Additionally, if you do not have an reverse phase encoded B0 image for distortion correction, you can contact me about how to edit the code. This will require you not only to comment out the functions for distortion correction, but also to change the coregistration algorithm which uses currently distortion correction information. The current pipeline will do all of the following preprocessing steps in the order listed:

  Converts Data to Float
  Distortion Correction using topup
  Eddy Current Correction using eddy_openmp (Optional GPU)
  Fits a tensor model using dtifit
  Models Crossing Fibers Using (Optional GPU)
  Coregisters DTI data using epi_reg
  Normalizes Data using fnirt
  Calculates inverse transform from MNI to DTI space, and structural to DTI Space. 

-------------------------------------------------------------------------------------------------------------
fMRI_Preprocessing_ACI.py
