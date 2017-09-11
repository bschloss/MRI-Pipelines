# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:23:15 2017

@author: bschloss
"""
import pickle as pkl
import numpy as np
from itertools import combinations
import argparse as ap
from scipy import io as sio
from sklearn.decomposition import TruncatedSVD as svd

parser = ap.ArgumentParser(description='Organize Data for Conceputal Change Regression Analysis')
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

words = [word.rstrip('\n') for word in open(''.join([pardir,'fMRI_Analyses/Conceptual_Change/fixwords.txt']),'r').readlines()]
wordpairs = list(combinations(words,2))
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
          
wordindex = pkl.load(open('/gpfs/group/pul8/default/read/BEAGLE/BEAGLEdata/indexList.pkl','r'))
contextmm = np.memmap('/gpfs/group/pul8/default/read/BEAGLE/BEAGLEdata/context', mode='r', dtype='float', shape=(len(wordindex),2000))
ordermm = np.memmap('/gpfs/group/pul8/default/read/BEAGLE/BEAGLEdata/order', mode='r', dtype='float', shape=(len(wordindex),2000))
w2v = pkl.load(open('/gpfs/group/pul8/default/read/Word2Vec/data.pkl','rb'))
sg=w2v['sg']
bow=w2v['bow']
sgbow=w2v['concat']

change_vecs = {}
avg_change = np.zeros((49596,),dtype=np.float32)
for i in range(len(words)):
    w = words[i]
    change_vecs[w] = pkl.load(open(''.join([pardir,'fMRI_Analyses/Conceptual_Change/',w.capitalize(),'/change_img.pkl']),'rb'))
    if i == 0:
        avg_change = np.zeros((change_vecs[w].shape[0],),dtype=np.float32)
    avg_change += change_vecs[w]
    
avg_change = avg_change/len(words)

sv = pkl.load(open(''.join([pardir,'fMRI_Analyses/Conceptual_Change/selected_voxels.pkl']),'rb'))
sv2_500 = np.asarray(pkl.load(open(''.join([pardir,'fMRI_Analyses/Conceptual_Change/selected_voxels2_500.pkl']),'rb')))
sv2_450 = np.asarray(pkl.load(open(''.join([pardir,'fMRI_Analyses/Conceptual_Change/selected_voxels2_450.pkl']),'rb')))
sv2_400 = np.asarray(pkl.load(open(''.join([pardir,'fMRI_Analyses/Conceptual_Change/selected_voxels2_400.pkl']),'rb')))
sv2_350 = np.asarray(pkl.load(open(''.join([pardir,'fMRI_Analyses/Conceptual_Change/selected_voxels2_350.pkl']),'rb')))
sv2_300 = np.asarray(pkl.load(open(''.join([pardir,'fMRI_Analyses/Conceptual_Change/selected_voxels2_300.pkl']),'rb')))
sv2_250 = np.asarray(pkl.load(open(''.join([pardir,'fMRI_Analyses/Conceptual_Change/selected_voxels2_250.pkl']),'rb')))
sv2_200 = np.asarray(pkl.load(open(''.join([pardir,'fMRI_Analyses/Conceptual_Change/selected_voxels2_200.pkl']),'rb')))
sv2_150 = np.asarray(pkl.load(open(''.join([pardir,'fMRI_Analyses/Conceptual_Change/selected_voxels2_150.pkl']),'rb')))
sv2_100 = np.asarray(pkl.load(open(''.join([pardir,'fMRI_Analyses/Conceptual_Change/selected_voxels2_100.pkl']),'rb')))
sv2_50 = np.asarray(pkl.load(open(''.join([pardir,'fMRI_Analyses/Conceptual_Change/selected_voxels2_50.pkl']),'rb')))

design_context = np.ndarray((len(wordpairs),len(words)-2,2000),dtype=np.float32)
design_order = np.ndarray((len(wordpairs),len(words)-2,2000),dtype=np.float32)
design_contextorder = np.ndarray((len(wordpairs),len(words)-2,4000),dtype=np.float32)
design_sg = np.ndarray((len(wordpairs),len(words)-2,2000),dtype=np.float32)
design_bow = np.ndarray((len(wordpairs),len(words)-2,2000),dtype=np.float32)
design_sgbow = np.ndarray((len(wordpairs),len(words)-2,4000),dtype=np.float32)
response = np.ndarray((len(wordpairs),len(words)-2,500),dtype=np.float32)
response2_500 = np.ndarray((len(wordpairs),len(words)-2,sv2_500.shape[0]),dtype=np.float32)
response2_450 = np.ndarray((len(wordpairs),len(words)-2,sv2_450.shape[0]),dtype=np.float32)
response2_400 = np.ndarray((len(wordpairs),len(words)-2,sv2_400.shape[0]),dtype=np.float32)
response2_350 = np.ndarray((len(wordpairs),len(words)-2,sv2_350.shape[0]),dtype=np.float32)
response2_300 = np.ndarray((len(wordpairs),len(words)-2,sv2_300.shape[0]),dtype=np.float32)
response2_250 = np.ndarray((len(wordpairs),len(words)-2,sv2_250.shape[0]),dtype=np.float32)
response2_200 = np.ndarray((len(wordpairs),len(words)-2,sv2_200.shape[0]),dtype=np.float32)
response2_150 = np.ndarray((len(wordpairs),len(words)-2,sv2_150.shape[0]),dtype=np.float32)
response2_100 = np.ndarray((len(wordpairs),len(words)-2,sv2_100.shape[0]),dtype=np.float32)
response2_50 = np.ndarray((len(wordpairs),len(words)-2,sv2_50.shape[0]),dtype=np.float32)

lo_context = np.ndarray((len(wordpairs),2,2000),dtype=np.float32)
lo_order = np.ndarray((len(wordpairs),2,2000),dtype=np.float32)
lo_contextorder = np.ndarray((len(wordpairs),2,4000),dtype=np.float32)
lo_sg = np.ndarray((len(wordpairs),2,2000),dtype=np.float32)
lo_bow = np.ndarray((len(wordpairs),2,2000),dtype=np.float32)
lo_sgbow = np.ndarray((len(wordpairs),2,4000),dtype=np.float32)
lo_response = np.ndarray((len(wordpairs),2,500),dtype=np.float32)
lo_response2_500 = np.ndarray((len(wordpairs),2,sv2_500.shape[0]),dtype=np.float32)
lo_response2_450 = np.ndarray((len(wordpairs),2,sv2_450.shape[0]),dtype=np.float32)
lo_response2_400 = np.ndarray((len(wordpairs),2,sv2_400.shape[0]),dtype=np.float32)
lo_response2_350 = np.ndarray((len(wordpairs),2,sv2_350.shape[0]),dtype=np.float32)
lo_response2_300 = np.ndarray((len(wordpairs),2,sv2_300.shape[0]),dtype=np.float32)
lo_response2_250 = np.ndarray((len(wordpairs),2,sv2_250.shape[0]),dtype=np.float32)
lo_response2_200 = np.ndarray((len(wordpairs),2,sv2_200.shape[0]),dtype=np.float32)
lo_response2_150 = np.ndarray((len(wordpairs),2,sv2_150.shape[0]),dtype=np.float32)
lo_response2_100 = np.ndarray((len(wordpairs),2,sv2_100.shape[0]),dtype=np.float32)
lo_response2_50 = np.ndarray((len(wordpairs),2,sv2_50.shape[0]),dtype=np.float32)

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
    lo_response2_500[i,0,:] = change_vecs[w1][sv2_500] - avg_change[sv2_500]
    lo_response2_500[i,1,:] = change_vecs[w2][sv2_500] - avg_change[sv2_500]
    lo_response2_450[i,0,:] = change_vecs[w1][sv2_450] - avg_change[sv2_450]
    lo_response2_450[i,1,:] = change_vecs[w2][sv2_450] - avg_change[sv2_450]
    lo_response2_400[i,0,:] = change_vecs[w1][sv2_400] - avg_change[sv2_400]
    lo_response2_400[i,1,:] = change_vecs[w2][sv2_400] - avg_change[sv2_400]
    lo_response2_350[i,0,:] = change_vecs[w1][sv2_350] - avg_change[sv2_350]
    lo_response2_350[i,1,:] = change_vecs[w2][sv2_350] - avg_change[sv2_350]
    lo_response2_300[i,0,:] = change_vecs[w1][sv2_300] - avg_change[sv2_300]
    lo_response2_300[i,1,:] = change_vecs[w2][sv2_300] - avg_change[sv2_300]
    lo_response2_250[i,0,:] = change_vecs[w1][sv2_250] - avg_change[sv2_250]
    lo_response2_250[i,1,:] = change_vecs[w2][sv2_250] - avg_change[sv2_250]
    lo_response2_200[i,0,:] = change_vecs[w1][sv2_200] - avg_change[sv2_200]
    lo_response2_200[i,1,:] = change_vecs[w2][sv2_200] - avg_change[sv2_200]
    lo_response2_150[i,0,:] = change_vecs[w1][sv2_150] - avg_change[sv2_150]
    lo_response2_150[i,1,:] = change_vecs[w2][sv2_150] - avg_change[sv2_150]
    lo_response2_100[i,0,:] = change_vecs[w1][sv2_100] - avg_change[sv2_100]
    lo_response2_100[i,1,:] = change_vecs[w2][sv2_100] - avg_change[sv2_100]
    lo_response2_50[i,0,:] = change_vecs[w1][sv2_50] - avg_change[sv2_50]
    lo_response2_50[i,1,:] = change_vecs[w2][sv2_50] - avg_change[sv2_50]
    rownum = 0
    for j in range(len(words)):
        word = words[j]
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
            response2_500[i,rownum,:] = change_vecs[word][sv2_500] - avg_change[sv2_500]
            response2_450[i,rownum,:] = change_vecs[word][sv2_450] - avg_change[sv2_450]
            response2_400[i,rownum,:] = change_vecs[word][sv2_400] - avg_change[sv2_400]
            response2_350[i,rownum,:] = change_vecs[word][sv2_350] - avg_change[sv2_350]
            response2_300[i,rownum,:] = change_vecs[word][sv2_300] - avg_change[sv2_300]
            response2_250[i,rownum,:] = change_vecs[word][sv2_250] - avg_change[sv2_250]
            response2_200[i,rownum,:] = change_vecs[word][sv2_200] - avg_change[sv2_200]
            response2_150[i,rownum,:] = change_vecs[word][sv2_150] - avg_change[sv2_150]
            response2_100[i,rownum,:] = change_vecs[word][sv2_100] - avg_change[sv2_100]
            response2_50[i,rownum,:] = change_vecs[word][sv2_50] - avg_change[sv2_50]
            rownum += 1
          
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/design_context.mat']),{'design_context' : design_context})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/design_order.mat']),{'design_order': design_order})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/design_contextorder.mat']),{'design_contextorder' : design_contextorder})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/design_sg.mat']),{'design_sg' : design_sg})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/design_bow.mat']),{'design_bow' : design_bow})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/design_sgbow.mat']),{'design_sgbow' : design_sgbow})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/response.mat']),{'response' : response})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/response2_500.mat']),{'response2_500' : response2_500})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/response2_450.mat']),{'response2_450' : response2_450})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/response2_400.mat']),{'response2_400' : response2_400})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/response2_350.mat']),{'response2_350' : response2_350})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/response2_300.mat']),{'response2_300' : response2_300})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/response2_250.mat']),{'response2_250' : response2_250})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/response2_200.mat']),{'response2_200' : response2_200})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/response2_150.mat']),{'response2_150' : response2_150})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/response2_100.mat']),{'response2_100' : response2_100})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/response2_50.mat']),{'response2_50' : response2_50})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_context.mat']),{'lo_context' : lo_context})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_order.mat']),{'lo_order': lo_order})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_contextorder.mat']),{'lo_contextorder' : lo_contextorder})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_sg.mat']),{'lo_sg' : lo_sg})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_bow.mat']),{'lo_bow' : lo_bow})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_sgbow.mat']),{'lo_sgbow' : lo_sgbow})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_response.mat']),{'lo_response' : lo_response})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_response2_500.mat']),{'lo_response2_500' : lo_response2_500})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_response2_450.mat']),{'lo_response2_450' : lo_response2_450})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_response2_400.mat']),{'lo_response2_400' : lo_response2_400})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_response2_350.mat']),{'lo_response2_350' : lo_response2_350})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_response2_300.mat']),{'lo_response2_300' : lo_response2_300})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_response2_250.mat']),{'lo_response2_250' : lo_response2_250})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_response2_200.mat']),{'lo_response2_200' : lo_response2_200})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_response2_150.mat']),{'lo_response2_150' : lo_response2_150})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_response2_100.mat']),{'lo_response2_100' : lo_response2_100})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_response2_50.mat']),{'lo_response2_50' : lo_response2_50})

dims = min(30,design_context.shape[1])
SVD = svd(n_components = dims,n_iter=100)
design_context_svd = np.ndarray((design_context.shape[0],design_context.shape[1],dims),dtype=np.float)
design_order_svd = np.ndarray((design_order.shape[0],design_order.shape[1],dims),dtype=np.float)
design_contextorder_svd = np.ndarray((design_contextorder.shape[0],design_contextorder.shape[1],dims),dtype=np.float)
design_sg_svd = np.ndarray((design_sg.shape[0],design_sg.shape[1],dims),dtype=np.float)
design_bow_svd = np.ndarray((design_bow.shape[0],design_bow.shape[1],dims),dtype=np.float)
design_sgbow_svd = np.ndarray((design_sgbow.shape[0],design_sgbow.shape[1],dims),dtype=np.float)
lo_context_svd = np.ndarray((lo_context.shape[0],lo_context.shape[1],dims),dtype=np.float)
lo_order_svd = np.ndarray((lo_order.shape[0],lo_order.shape[1],dims),dtype=np.float)
lo_contextorder_svd = np.ndarray((lo_contextorder.shape[0],lo_contextorder.shape[1],dims),dtype=np.float)
lo_sg_svd = np.ndarray((lo_sg.shape[0],lo_sg.shape[1],dims),dtype=np.float)
lo_bow_svd = np.ndarray((lo_bow.shape[0],lo_bow.shape[1],dims),dtype=np.float)
lo_sgbow_svd = np.ndarray((lo_sgbow.shape[0],lo_sgbow.shape[1],dims),dtype=np.float)
lo_context_svd = np.ndarray((lo_context.shape[0],lo_context.shape[1],dims),dtype=np.float)
lo_order_svd = np.ndarray((lo_order.shape[0],lo_order.shape[1],dims),dtype=np.float)

for i in range(design_context.shape[0]):
    SVD.fit(design_context[i,:,:])
    design_context_svd[i,:,:] = SVD.transform(design_context[i,:,:])
    lo_context_svd[i,:,:] = SVD.transform(lo_context[i,:,:])
    SVD.fit(design_order[i,:,:])
    design_order_svd[i,:,:] = SVD.transform(design_order[i,:,:])
    lo_order_svd[i,:,:] = SVD.transform(lo_order[i,:,:])
    SVD.fit(design_contextorder[i,:,:])
    design_contextorder_svd[i,:,:] = SVD.transform(design_contextorder[i,:,:])
    lo_contextorder_svd[i,:,:] = SVD.transform(lo_contextorder[i,:,:])
    SVD.fit(design_sg[i,:,:])
    design_sg_svd[i,:,:] = SVD.transform(design_sg[i,:,:])
    lo_sg_svd[i,:,:] = SVD.transform(lo_sg[i,:,:])
    SVD.fit(design_bow[i,:,:])
    design_bow_svd[i,:,:] = SVD.transform(design_bow[i,:,:])
    lo_bow_svd[i,:,:] = SVD.transform(lo_bow[i,:,:])
    SVD.fit(design_sgbow[i,:,:])
    design_sgbow_svd[i,:,:] = SVD.transform(design_sgbow[i,:,:])
    lo_sgbow_svd[i,:,:] = SVD.transform(lo_sgbow[i,:,:]) 
    
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/design_context_svd.mat']),{'design_context_svd' : design_context_svd})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/design_order_svd.mat']),{'design_order_svd': design_order_svd})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/design_contextorder_svd.mat']),{'design_contextorder_svd' : design_contextorder_svd})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/design_sg_svd.mat']),{'design_sg_svd' : design_sg_svd})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/design_bow_svd.mat']),{'design_bow_svd' : design_bow_svd})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/design_sgbow_svd.mat']),{'design_sgbow_svd' : design_sgbow_svd})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_context_svd.mat']),{'lo_context_svd' : lo_context_svd})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_order_svd.mat']),{'lo_order_svd': lo_order_svd})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_contextorder_svd.mat']),{'lo_contextorder_svd' : lo_contextorder_svd})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_sg_svd.mat']),{'lo_sg_svd' : lo_sg_svd})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_bow_svd.mat']),{'lo_bow_svd' : lo_bow_svd})
sio.savemat(''.join([pardir,'fMRI_Analyses/Conceptual_Change/lo_sgbow_svd.mat']),{'lo_sgbow_svd' : lo_sgbow_svd})