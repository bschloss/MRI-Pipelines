# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import nibabel as nib
import numpy as np
import pickle as pkl
from itertools import combinations

target = ['gps',
          'mars',
          'axial',
          'battery',
          'circuit',
          'combination',
          'current',
          'distance',
          'earth',
          'electric',
          'electron',
          'gravity',
          'item',
          'lightbulb',
          'live',
          'location',
          'number',
          'ocean',
          'oil',
          'orbit',
          'order',
          'path',
          'permutation',
          'radio',
          'receiver',
          'rover',
          'safety',
          'satellite',
          'scientist',
          'select',
          'set',
          'signal',
          'source',
          'space',
          'spill',
          'temperature',
          'tilt',
          'time',
          'wire']
        
target = [word.capitalize() for word in target]   
datadir = '/gpfs/group/pul8/default/read/'
pardirs = [datadir + par + '/' for par in ['201','002','003','004','105','006','107','008','009','110',
                                           '011','012','013',      '015','016','017','018','019','020',
                                           '021','122','023','024','025','026','027','028','029','030',
                                           '031',      '033','034','035','036','037','038','039','040',
                                           '041','042','043']]



change = {} 
for tar in target:
    change[tar] = np.ndarray((60*73*46,1),dtype=np.float)
    rownum = 0
    for i in range(len(pardirs)):
        pardir = pardirs[i]
        tardir = pardir + 'fMRI_Analyses/Conceptual_Change/' + tar
        imgs = []
        if os.path.isdir(''.join([tardir,str(1)])):
            imgs = imgs + sorted([tardir + '1/Target_Copes/_Target_Fit_Model0/' + cope for cope in os.listdir(''.join([tardir,str(1),'/Target_Copes/_Target_Fit_Model0/'])) if ('cope1.' not in cope and 'cope2.' not in cope)])
        if os.path.isdir(''.join([tardir,str(2)])):
            imgs = imgs + sorted([tardir + '2/Target_Copes/_Target_Fit_Model0/' + cope for cope in os.listdir(''.join([tardir,str(2),'/Target_Copes/_Target_Fit_Model0/'])) if ('cope1.' not in cope and 'cope2.' not in cope)])
        if len(imgs)>1:
            if rownum == 0:
                change[tar][:,rownum] = np.ravel(nib.load(imgs[-1]).get_data()) - np.ravel(nib.load(imgs[0]).get_data())
                rownum += 1
            else:
                change[tar] = np.concatenate((change[tar],(np.ravel(nib.load(imgs[-1]).get_data()) - np.ravel(nib.load(imgs[0]).get_data())).reshape((60*73*46,1))),axis=1) 

vox_ind = []
for i in range(change[tar].shape[0]):
    if sum([sum(change[tar][i,:]==float(0)) for tar in change.keys()]) == 0:
        vox_ind.append(i)
vox_ind = np.asarray(vox_ind)
for tar in change.keys():
    change[tar] = change[tar][vox_ind,:]
    pkl.dump(np.mean(change[tar],axis=1),open(''.join(['/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/avg_',tar.lower(),'_change.pkl']),'wb'))
    np.savetxt(''.join(['/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/avg_',tar.lower(),'_change.txt']),np.mean(change[tar],axis=1))

wordpairs = list(combinations(change.keys(),2))
selected_voxels = np.ndarray((len(wordpairs),500),dtype=np.int)
min_par = min([change[tar].shape[1] for tar in change.keys()])
trialnum = 0
for w1,w2 in wordpairs:
    words = [change.keys()[i] for i in range(len(change.keys())) if (change.keys()[i].lower() not in  [w1.lower(),w2.lower()])]
    data = np.ndarray((len(vox_ind),min_par,len(words)),dtype=np.float)  
    for i in range(len(words)):
        data[:,:,i] = change[words[i]][:,:min_par]
    vox_pairwise_corrs = np.ndarray((data.shape[0],1),dtype=np.float)
    for i in range(data.shape[0]):
        vox_pairwise_corrs[i] = (sum(sum(np.corrcoef(data[i,:,:])))-min_par)/2
    minimum = np.sort(vox_pairwise_corrs,axis=0)[-500][0]
    voxnum = 0
    for i in range(vox_pairwise_corrs.shape[0]):
        if vox_pairwise_corrs[i,0] >= minimum:
            selected_voxels[trialnum,voxnum] = i
            voxnum+=1
    trialnum += 1
    
fname = '/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/selected_voxels.pkl'
pkl.dump(selected_voxels,open(fname,'wb'))
np.savetxt(fname.replace('pkl','txt'),selected_voxels)       