# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:59:31 2017

@author: bschloss
"""

import os
import numpy as np
import nibabel as nib
import pickle as pkl
from sklearn.model_selection import LeaveOneOut
from sklearn import svm
from statsmodels.sandbox.stats.runs import mcnemar

datadir = '/gpfs/group/pul8/default/read/'

pars = ['201','002','003','004','105','006','107','008','009','110',
	   '011','012','013',      '015','016','017','018','019','020',
	   '021','122','023','024','025','026','027','028','029','030',
	   '031',      '033','034','035','036','037','038','039','040',
	   '041','042','043']

concrete = {'Battery':[],'Earth':[],'Lightbulb':[],'Oil':[],
            'Path':[],'Radio':[],'Satellite':[],'Scientist':[],
            'Pasta':[],'Pizza':[]}

abstract = {'Gps':[],'Circuit':[],'Combination':[],'Current':[],
            'Electric':[],'Order':[],'Permutation':[],'Signal':[],
            'Space':[],'Time':[]}

for word in concrete.keys():
    for par in pars:
        imgs = []
        worddir = ''.join([datadir,par,'/fMRI_Analyses/Conceptual_Change/',word,'1'])
        if os.path.isdir(worddir):
            copenum = 3
            while(os.path.isfile(''.join([worddir,'/Target_Copes/_Target_Fit_Model0/cope',str(copenum),'.nii.gz']))):
                imgs.append(''.join([worddir,'/Target_Copes/_Target_Fit_Model0/cope',str(copenum),'.nii.gz']))
                copenum += 1
        worddir = ''.join([datadir,par,'/fMRI_Analyses/Conceptual_Change/',word,'2'])
        if os.path.isdir(worddir):
            copenum = 3
            while(os.path.isfile(''.join([worddir,'/Target_Copes/_Target_Fit_Model0/cope',str(copenum),'.nii.gz']))):
                imgs.append(''.join([worddir,'/Target_Copes/_Target_Fit_Model0/cope',str(copenum),'.nii.gz']))
                copenum+=1
        if len(imgs)>3:
            imgs = imgs[:4]
            concrete[word].append(imgs)
    concrete[word] = concrete[word][:30]
                
for word in abstract.keys():
    for par in pars:
        imgs = []
        worddir = ''.join([datadir,par,'/fMRI_Analyses/Conceptual_Change/',word,'1'])
        if os.path.isdir(worddir):
            copenum = 3
            while(os.path.isfile(''.join([worddir,'/Target_Copes/_Target_Fit_Model0/cope',str(copenum),'.nii.gz']))):
                imgs.append(''.join([worddir,'/Target_Copes/_Target_Fit_Model0/cope',str(copenum),'.nii.gz']))
                copenum += 1
        worddir = ''.join([datadir,par,'/fMRI_Analyses/Conceptual_Change/',word,'2'])
        if os.path.isdir(worddir):
            copenum = 3
            while(os.path.isfile(''.join([worddir,'/Target_Copes/_Target_Fit_Model0/cope',str(copenum),'.nii.gz']))):
                imgs.append(''.join([worddir,'/Target_Copes/_Target_Fit_Model0/cope',str(copenum),'.nii.gz']))
                copenum+=1
        if len(imgs)>3:
            imgs = imgs[:4]
            abstract[word].append(imgs) 
    abstract[word] = abstract[word][:30]

   
mask = np.ravel(nib.load('/gpfs/group/pul8/default/read/ELN_Masks/LH_LN.nii.gz').get_data()).astype(dtype=int)

for word in concrete.keys():
    for par in range(30):
        for im in range(4):
            img = nib.load(concrete[word][par][im])
            concrete[word][par][im] = img.get_data()[mask] 
            img.uncache() 
for word in abstract.keys():
    for par in range(30):
        for im in range(4):
            img = nib.load(abstract[word][par][im])
            abstract[word][par][im] = img.get_data()[mask]  
  	    img.uncache()
pkl.dump(concrete,open('/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/concrete.pkl','wb'))
pkl.dump(abstract,open('/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/abstract.pkl','wb'))

word2num = {'Battery'       :   0,
            'Earth'         :   1,
            'Lightbul'      :   2,
            'Oil'           :   3,
            'Path'          :   4,
            'Radio'         :   5,
            'Sattelite'     :   6,
            'Scientist'     :   7,
            'Pasta'         :   8,
            'Pizza'         :   9,
            'Gps'           :   10,
            'Circuit'       :   11,
            'Combination'   :   12,
            'Current'       :   13,
            'Electric'      :   14,
            'Order'         :   15,
            'Permutation'   :   16,
            'Signal'        :   17,
            'Space'         :   18,
            'Time'          :   19}           
