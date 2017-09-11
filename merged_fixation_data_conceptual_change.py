# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import nibabel as nib
import numpy as np
import exceptions
import pickle as pkl
from itertools import combinations
import argparse as ap

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
#Define the directories where data is located
parser = ap.ArgumentParser(description='Preprocess DTI data and put in new folder')
parser.add_argument('pardir', metavar="stanford", type=str,
                    help="Path to participant's directory")                                                  
args = parser.parse_args()
pardir= '/gpfs/group/pul8/default/read/'
if len(args.pardir) == 1:
    pardir += '00' + args.pardir +'/'
elif len(args.pardir) == 2:
    pardir += '0' + args.pardir + '/'
else:
    pardir += args.pardir + '/'

parwords = ''
voxel_selection_index = []
total = 0

for tar in target:
    tardir = pardir + 'fMRI_Analyses/Conceptual_Change_ID/' + tar
    imgs = []
    if os.path.isdir(''.join([tardir,str(1)])):
        imgs = imgs + [tardir + '1/Parametric_Param_Estimates/_Parametric_Model_Fit0/' + pe for pe in sorted(os.listdir(''.join([tardir,'1/Parametric_Param_Estimates/_Parametric_Model_Fit0/'])))[7:]]
    if os.path.isdir(''.join([tardir,str(2)])):
        imgs = imgs + [tardir + '2/Parametric_Param_Estimates/_Parametric_Model_Fit0/' + pe for pe in sorted(os.listdir(''.join([tardir,'2/Parametric_Param_Estimates/_Parametric_Model_Fit0/'])))[7:]]
    if len(imgs) >= 2:   
        parwords += tar.lower() + '\n'
        total += len(imgs)
        if len(imgs) >= 3: 
            voxel_selection_index.append(1)
        else:
            voxel_selection_index.append(0)
        d = np.ndarray((60*73*46,len(imgs)),dtype=np.float)
        for i in range(len(imgs)):
            d[:,i] = np.ravel(nib.load(imgs[i]).get_data())
        try:
            os.makedirs(tardir)
        except:
            exceptions.OSError
        pkl.dump(d,open('/'.join([tardir,'fixdata.pkl']),'wb'))
        
open(''.join([pardir,'fMRI_Analyses/Conceptual_Change_ID/fixwords.txt']),'w').write(parwords.rstrip('\n'))
pkl.dump(voxel_selection_index,open(''.join([pardir,'fMRI_Analyses/Conceptual_Change_ID/voxel_selection_indeces.pkl']),'wb'))

alles = np.ndarray((60*73*46,total),dtype=np.float)  
imnum = 0
for tar in [word.capitalize() for word in parwords.split()]:
    tardir = pardir + 'fMRI_Analyses/Conceptual_Change_ID/' + tar
    d = pkl.load(open('/'.join([tardir,'fixdata.pkl']),'rb'))
    for i in range(d.shape[1]):
        alles[:,imnum] = d[:,i]
        imnum += 1
        
vox_ind = []
for i in range(alles.shape[0]):
    if sum(alles[i,:]==float(0)) == 0:
        vox_ind.append(i)
vox_ind = np.asarray(vox_ind)
del alles

coords = np.ndarray((60,73,46),dtype='|S6')
for i in range(coords.shape[0]):
    s0 = str(i)
    if len(s0)==1:
        s0 = '0' + s0
    for j in range(coords.shape[1]):
        s1 = str(j)
        if len(s1)==1:
            s1 = '0' + s1
        for k in range(coords.shape[2]):
            s2 = str(k)
            if len(s2)==1:
                s2 = '0' + s2
            coords[i,j,k] = s0 + s1 + s2
coords = coords.ravel()[vox_ind]  
pkl.dump(coords,open(''.join([pardir,'fMRI_Analyses/Conceptual_Change_ID/voxel_coords.pkl']),'wb')) 

for tar in [word.capitalize() for word in parwords.split()]:
    tardir = pardir + 'fMRI_Analyses/Conceptual_Change_ID/' + tar
    d = pkl.load(open('/'.join([tardir,'fixdata.pkl']),'rb'))[vox_ind,:]
    pkl.dump(d,open('/'.join([tardir,'fixdata.pkl']),'wb'))
    change_img = np.subtract(d[:,-1],d[:,0])
    np.savetxt('/'.join([tardir,'change_img.txt']),change_img)
    pkl.dump(change_img,open('/'.join([tardir,'change_img.pkl']),'wb'))

alles = {} 
del d
del change_img

for i in range(len([word.capitalize() for word in parwords.split()])):
    tar = [word.capitalize() for word in parwords.split()][i]
    tardir = pardir + 'fMRI_Analyses/Conceptual_Change_ID/' + tar
    alles[tar.lower()] = pkl.load(open('/'.join([tardir,'fixdata.pkl']),'rb'))[:,:3] 

wordpairs = list(combinations(parwords.split(),2))
selected_voxels = np.ndarray((len(wordpairs),500),dtype=np.int)
trialnum = 0
for w1,w2 in wordpairs:
    words = [parwords.split()[i] for i in range(len(parwords.split())) if (voxel_selection_index[i] == 1 and parwords.split()[i].lower() not in  [w1.lower(),w2.lower()])]
    data = np.ndarray((len(vox_ind),3,len(words)),dtype=np.float)  
    for i in range(len(words)):
        data[:,:,i] = alles[words[i]]
    vox_pairwise_corrs = np.ndarray((data.shape[0],1),dtype=np.float)
    for i in range(data.shape[0]):
        vox_pairwise_corrs[i] = (sum(sum(np.corrcoef(data[i,:,:])))-3)/2
    minimum = np.sort(vox_pairwise_corrs,axis=0)[-500][0]
    voxnum = 0
    for i in range(vox_pairwise_corrs.shape[0]):
        if vox_pairwise_corrs[i,0] >= minimum:
            selected_voxels[trialnum,voxnum] = i
            voxnum+=1
    trialnum+=1
fname = pardir + 'fMRI_Analyses/Conceptual_Change_ID/selected_voxels.pkl'
pkl.dump(selected_voxels,open(fname,'wb'))
np.savetxt(fname.replace('pkl','txt'),selected_voxels)       
