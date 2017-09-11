# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:23:15 2017

@author: bschloss
"""
import pickle as pkl
import numpy as np
from itertools import combinations
from scipy import io as sio

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
wordpairs = list(combinations(target,2))         
wordindex = pkl.load(open('/gpfs/group/pul8/default/read/BEAGLE/BEAGLEdata/indexList.pkl','r'))
contextmm = np.memmap('/gpfs/group/pul8/default/read/BEAGLE/BEAGLEdata/context', mode='r', dtype='float', shape=(len(wordindex),2000))
ordermm = np.memmap('/gpfs/group/pul8/default/read/BEAGLE/BEAGLEdata/order', mode='r', dtype='float', shape=(len(wordindex),2000))
w2v = pkl.load(open('/gpfs/group/pul8/default/read/Word2Vec/data.pkl','rb'))
sg=w2v['sg']
bow=w2v['bow']
sgbow=w2v['concat']

change_vecs = {}
avg_change = np.zeros((49596,),dtype=np.float32)
for i in range(len(target)):
    w = target[i]
    change_vecs[w] = pkl.load(open('_'.join(['/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/avg',w,'change.pkl']),'rb'))
    if i == 0:
        avg_change = np.zeros((change_vecs[w].shape[0],),dtype=np.float32)
    avg_change += change_vecs[w]
    
avg_change = avg_change/len(target)
sv = pkl.load(open('/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/selected_voxels.pkl','rb'))

design_context = np.ndarray((len(wordpairs),len(target)-2,2000),dtype=np.float32)
design_order = np.ndarray((len(wordpairs),len(target)-2,2000),dtype=np.float32)
design_contextorder = np.ndarray((len(wordpairs),len(target)-2,4000),dtype=np.float32)
design_sg = np.ndarray((len(wordpairs),len(target)-2,2000),dtype=np.float32)
design_bow = np.ndarray((len(wordpairs),len(target)-2,2000),dtype=np.float32)
design_sgbow = np.ndarray((len(wordpairs),len(target)-2,4000),dtype=np.float32)
response = np.ndarray((len(wordpairs),len(target)-2,500),dtype=np.float32)

lo_context = np.ndarray((len(wordpairs),2,2000),dtype=np.float32)
lo_order = np.ndarray((len(wordpairs),2,2000),dtype=np.float32)
lo_contextorder = np.ndarray((len(wordpairs),2,4000),dtype=np.float32)
lo_sg = np.ndarray((len(wordpairs),2,2000),dtype=np.float32)
lo_bow = np.ndarray((len(wordpairs),2,2000),dtype=np.float32)
lo_sgbow = np.ndarray((len(wordpairs),2,4000),dtype=np.float32)
lo_response = np.ndarray((len(wordpairs),2,500),dtype=np.float32)

for i in range(len(wordpairs)):
    w1 = wordpairs[i][0]
    w2 = wordpairs[i][1]
    lo_context[i,0,:] = contextmm[wordindex[w1]]
    lo_context[i,1,:] = contextmm[wordindex[w2]]
    lo_order[i,0,:] = ordermm[wordindex[w1]]
    lo_order[i,1,:] = ordermm[wordindex[w2]]
    lo_contextorder[i,0,:2000] = contextmm[wordindex[w1]]
    lo_contextorder[i,1,:2000] = contextmm[wordindex[w2]]
    lo_contextorder[i,0,2000:] = ordermm[wordindex[w1]]
    lo_contextorder[i,1,2000:] = ordermm[wordindex[w2]]
    lo_sg[i,0,:] = sg[:,target.index(w1)]
    lo_sg[i,1,:] = sg[:,target.index(w2)]
    lo_bow[i,0,:] = bow[:,target.index(w1)]
    lo_bow[i,1,:] = bow[:,target.index(w2)]
    lo_sgbow[i,0,:2000] = sg[:,target.index(w1)]
    lo_sgbow[i,1,:2000] = sg[:,target.index(w2)]
    lo_sgbow[i,0,2000:] = bow[:,target.index(w1)]
    lo_sgbow[i,1,2000:] = bow[:,target.index(w2)]
    lo_response[i,0,:] = change_vecs[w1][sv[i,:]] - avg_change[sv[i,:]]
    lo_response[i,1,:] = change_vecs[w2][sv[i,:]] - avg_change[sv[i,:]]
    rownum = 0
    for j in range(len(target)):
        word = target[j]
        if word != w1 and word != w2:
            design_context[i,rownum,:] = contextmm[wordindex[word]]
            design_order[i,rownum,:] = ordermm[wordindex[word]]
            design_contextorder[i,rownum,:2000] = contextmm[wordindex[word]]
            design_contextorder[i,rownum,2000:] = ordermm[wordindex[word]]
            design_sg[i,rownum,:] = sg[:,target.index(word)]
            design_bow[i,rownum,:] = bow[:,target.index(word)]
            design_sgbow[i,rownum,:2000] = sg[:,target.index(word)]
            design_sgbow[i,rownum,2000:] = bow[:,target.index(word)]
            response[i,rownum,:] = change_vecs[word][sv[i,:]] - avg_change[sv[i,:]]
            rownum += 1
            
sio.savemat('/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/design_context.mat',{'design_context' : design_context})
sio.savemat('/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/design_order.mat',{'design_order': design_order})
sio.savemat('/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/design_contextorder.mat',{'design_contextorder' : design_contextorder})
sio.savemat('/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/design_sg.mat',{'design_sg' : design_sg})
sio.savemat('/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/design_bow.mat',{'design_bow' : design_bow})
sio.savemat('/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/design_sgbow.mat',{'design_sgbow' : design_sgbow})
sio.savemat('/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/response.mat',{'response' : response})
sio.savemat('/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/lo_context.mat',{'lo_context' : lo_context})
sio.savemat('/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/lo_order.mat',{'lo_order': lo_order})
sio.savemat('/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/lo_contextorder.mat',{'lo_contextorder' : lo_contextorder})
sio.savemat('/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/lo_sg.mat',{'lo_sg' : lo_sg})
sio.savemat('/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/lo_bow.mat',{'lo_bow' : lo_bow})
sio.savemat('/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/lo_sgbow.mat',{'lo_sgbow' : lo_sgbow})
sio.savemat('/gpfs/group/pul8/default/read/Group_Analyses/Conceptual_Change/lo_response.mat',{'lo_response' : lo_response})
