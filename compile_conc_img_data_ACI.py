# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 00:12:36 2017

@author: bschloss
"""
import os
import nibabel as nib
import numpy as np
import exceptions
import pickle as pkl
conc = ['battery',
        'boiler',
        'building',
        'canyon',
        'cheese',
        'circle',
        'coast',
        'dust',
        #'earth',
        'engineer',
        #'ground',
        'hull',
        'human',
        'instrument',
        'lightbulb',
        'loop',
        'ocean',
        'oil',
        'pasta',
        'path',
        'pepperoni',
        #'piece',
        'pizza',
        #'plant',
        'propeller',
        'radio',
        'rice',
        'satellite',
        'sausage',
        'scientist',
        'ship',
        'shore',
        'spacecraft',
        'station',
        'storm',
        'sun',
        'wave',
        'wire']#,
        #'world']
        
conc = [word.capitalize() for word in conc]   
    
regions = ['L_Angular_Gyrus',
           'L_Brocas_Area',
           'L_Inferior_Temporal_Gyrus',
           'L_Lateral_Occipital_Cortex',
           'L_Middle_Temporal_Gyrus',
           'L_Precuneous_and_L_Posterior_Cingulate',
           'L_Primary_Motor_Cortex',
           'L_Superior_Frontal_Gyrus',
           'L_Superior_Temporal_Gyrus',
           'L_Supramarginal_Gyrus',
           'L_Temporal_Fusiform_Cortex',
           'L_Visual_Cortex']

direc = '/gpfs/group/pul8/default/read/Group_Analyses/Concrete_Nouns_Bilingual/Compiled_Data_by_Region/'
try:
    os.makedirs(direc)
except:
    exceptions.OSError
data = {}                       
for region in regions:
    d = np.ndarray((60,73,46,len(conc)),dtype=np.float)
    for i in range(len(conc)):
        word = conc[i]
        img = '/gpfs/group/pul8/default/read/Group_Analyses/Concrete_Nouns/' + word + '/What_Path_Masked_Copes/' + region + '/'
        img += os.listdir(img)[0]
        img += '/' + os.listdir(img)[0]
        d[:,:,:,i] = nib.load(img).get_data()
    d = d.reshape((60*73*46,len(conc)))
    new = [d[i,:] for i in range(len(d)) if sum(d[i,:]==float(0)) == 0]
    data[region] = np.ndarray((len(new),len(conc)),dtype=np.float)
    for i in range(len(new)):
        data[region][i,:] = new[i]
pkl.dump(data,open(''.join([direc,'data_voxel_by_words_by_what_path_region.pkl']),'wb'))

conc = ['battery',
        'boiler',
        'building',
        'canyon',
        'cheese',
        'circle',
        'coast',
        'dust',
        'earth',
        'engineer',
        'ground',
        'hull',
        'human',
        'instrument',
        'lightbulb',
        'loop',
        'ocean',
        'oil',
        'pasta',
        'path',
        'pepperoni',
        'piece',
        'pizza',
        'plant',
        'propeller',
        'radio',
        'rice',
        'satellite',
        'sausage',
        'scientist',
        'ship',
        'shore',
        'spacecraft',
        'station',
        'storm',
        'sun',
        'wave',
        'wire',
        'world']

conc = [word.capitalize() for word in conc]

regions = ['LIFG',
           'RIFG',
           'LSTG',
           'RSTG',
           'LMTG',
           'RMTG',
           'LTP',
           'RTP']

data = {}                       
for region in regions:
    d = np.ndarray((60,73,46,len(conc)),dtype=np.float)
    for i in range(len(conc)):
        word = conc[i]
        img = '/gpfs/group/pul8/default/read/Group_Analyses/Concrete_Nouns_Bilingual/' + word + '/ELN_Masked_Copes/' + region + '/'
        img += os.listdir(img)[0]
        img += '/' + os.listdir(img)[0]
        d[:,:,:,i] = nib.load(img).get_data()
    d = d.reshape((60*73*46,len(conc)))
    new = [d[i,:] for i in range(len(d)) if sum(d[i,:]==float(0)) == 0]
    data[region] = np.ndarray((len(new),len(conc)),dtype=np.float)
    for i in range(len(new)):
        data[region][i,:] = new[i]
pkl.dump(data,open(''.join([direc,'data_voxel_by_words_by_eln_region.pkl']),'wb'))        
