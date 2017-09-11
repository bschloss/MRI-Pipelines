# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 03:33:14 2017

@author: bschloss
"""
import nibabel as nib
import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt
from statsmodels.api import OLS as OLS
import matplotlib.lines as mlines

def normalize(arr):
    numsonly = [val for val in arr if val != '???']
    mean = np.mean(np.asarray(numsonly))
    std = np.std(np.asarray(numsonly))
    for i in range(len(arr)):
        if arr[i] != '???':
            if abs((arr[i] - mean)/std) < std*3:
                arr[i] = (arr[i] - mean)/std
            else:
                arr[i] = 0.0
        else:
            arr[i] = 0.0
    return np.asarray(arr).reshape(len(arr),1)

def get_pvals(summary):
    s = str(summary).split()
    ps = []
    i = 0
    while i< len(s):
        if 'const' == s[i]:
            ps.append(' '.join(['Intercept',''.join(['p=',s[i+4]])]))
        if 'x' == s[i][0]:
            if '1' in s[i]:
                ps.append(' '.join(['GSRT',''.join(['p=',s[i+4]])]))
            #if '2' in s[i]:
            #    ps.append('  '.join(['GSRT',''.join(['p=',s[i+4]])]))
        i += 1
    return ps
    
def get_coefs(summary):
    s = str(summary).split()
    ps = []
    i = 0
    while i< len(s):
        if 'const' == s[i]:
            ps.append(float(s[i+1]))
        if 'x' == s[i][0]:
            if '1' in s[i]:
                ps.append(float(s[i+1]))
            #if '2' in s[i]:
            #    ps.append(float(s[i+1]))
        i += 1
    return ps
    
datadir = '/gpfs/group/pul8/default/read/'
pardirs = ['201','002','003','004','105','006','107','008','009','110',
           '011','012','013','214','015','016','017','018','019','020',
           '021','122','023','024','025','026','027','028','029','030',
           '031','132','033','034','035','036','037','038','039','040',
           '041','042','043','044','045','046','047','048','049','050']
runs = ['0','1','2','3','4']
regions = ['DCN','lCC','lFEF','mFEF','pons']
num2avgwl = {'1':5.56315789474,
           '2':6.13917525773,
           '3':6.37062937063,
           '4':6.08074534161,
           '5':6.1975308642}
rp = [line.rstrip('\r\n') for line in open('/home/bschloss/pul8_read/Data/randperm.txt','r').readlines()]           
data = {}
for par in pardirs:
    data[par] = {}
    for r in runs:
        if not (par == '021' and r == '4'):
            data[par][r] = {}
            for region in regions:
                f = datadir + par + '/fMRI_Analyses/Low_Level_Quad/Length_Vertex/_Parametric_Fit_Model' + r + '/vertex_x_val_' + region + '.nii.gz'
                data[par][r][region] = np.mean([val for val in list(nib.load(f).get_data().ravel()) if val != 0.0])
os.mkdir('/gpfs/group/pul8/default/read/Group_Analyses/Low_Level_Quad/Length_Vertex')
pkl.dump(data,open('/gpfs/group/pul8/default/read/Group_Analyses/Low_Level_Quad/Length_Vertex/vertex_data.pkl','wb'))
#bd = pkl.load(open('/gpfs/group/pul8/default/read/Group_Analyses/behavioral_data.pkl','rb'))
bd = pkl.load(open('/home/bschloss/pul8_read/MRI/Data/behavioral_data.pkl','rb'))
#data = pkl.load(open('/gpfs/group/pul8/default/read/Group_Analyses/Low_Level_Quad/Length_Vertex/vertex_data.pkl','rb'))
data = pkl.load(open('/home/bschloss/pul8_read/MRI/Data/Length_Vertex/Length_Vertex.pkl','rb'))
for region in regions:
    rd = []
    gsrt = []
    for par in pardirs:
        if not (par == '021'):
            rd.append(np.mean([data[par][r][region] for r in runs]))
            gsrt.append(bd[par]['GSRT'])
        else:
            rd.append(np.mean([data[par][r][region] for r in runs[:-1]]))
            gsrt.append(bd[par]['GSRT'])
    plt.hist(rd)
    plt.title(' '.join([region.capitalize(),'Vertex Histogram']))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    #plt.savefig(''.join(['/gpfs/group/pul8/default/read/Group_Analyses/Low_Level_Quad/Length_Vertex/',region,'_histogram.png']))
    plt.savefig(''.join(['/home/bschloss/pul8_read/MRI/Data/Length_Vertex/',region,'_histogram.png']))
    plt.clf()
    
    meangsrt = np.mean([el for el in gsrt if el != '???'])
    meanrd = np.mean(rd)
    stdgsrt = np.std([el for el in gsrt if el != '???'])
    stdrd = np.std(rd)
    
    
    remove = []
    for i in range(len(gsrt)):
        if gsrt[i] == '???':
            remove.append(i)
        elif abs(gsrt[i] - meangsrt) > 2*stdgsrt:
            remove.append(i)
        elif abs(rd[i] - meanrd) > 2*stdrd:
            remove.append(i)
    for el in sorted(remove,reverse=True):
        gsrt.pop(el)
        rd.pop(el)
    x = np.reshape(np.linspace(min(rd),max(rd),1000),(1000,1))
    #print region
    #print normalize(rd)
    model = OLS(np.asarray(gsrt).reshape(len(gsrt),1),np.concatenate((np.ones((len(rd),1),dtype=np.float),np.asarray(rd).reshape((len(rd),1))),axis=1))
    open(''.join(['/home/bschloss/pul8_read/MRI/Data/Length_Vertex/',region,'_gsrt_vertex_regression.txt']),'w').write(str(model.fit().summary()))
    coefs = get_coefs(model.fit().summary())
    lbls = get_pvals(model.fit().summary())
    cyan_line = mlines.Line2D([], [], color='cyan',
                          label=lbls[1],lw=3.0)  
    plt.scatter(rd,gsrt, c='black', s=20, marker=".")
    plt.plot(x, coefs[0] + coefs[1]*x, color='cyan',
         linewidth=3) 
    plt.ylabel('GSRT')
    plt.xlabel('Vertex X Value')
    plt.title(region)              
    plt.legend(handles=[cyan_line],loc=9,prop={'size':12})
    plt.savefig(''.join(['/home/bschloss/pul8_read/MRI/Data/Length_Vertex/',region,'_gsrt_vertex_regression.png']))
    plt.clf()
        
    
    rd = []
    avg_wl = [] 
    wm = []
    gsrt = []
    for par in pardirs:
        if not (par == '021'):
            for r in runs:
                rd.append(data[par][r][region])
                avg_wl.append(num2avgwl[rp[5*(int(par[-2:])-1) + int(r)]])
                gsrt.append(bd[par]['GSRT'])
        else:
            for r in runs[:-1]:
                rd.append(data[par][r][region])
                avg_wl.append(num2avgwl[rp[5*(int(par[-2:])-1) + int(r)]])
                gsrt.append(bd[par]['GSRT'])
                
    x = np.reshape(np.linspace(min(rd),max(rd),1000),(1000,1))
    model = OLS(rd,np.concatenate((np.ones((len(rd),1),dtype=np.float),normalize(avg_wl),normalize(gsrt)),axis=1))
    coefs = get_coefs(model.fit().summary())
    lbls = get_pvals(model.fit().summary())
    cyan_line = mlines.Line2D([], [], color='cyan',
                          label=lbls[0],lw=3.0)  
    red_line = mlines.Line2D([], [], color='red',
                          label=lbls[1],lw=3.0) 
    plt.scatter(normalize(rd), normalize(gsrt), c='black', s=20, marker=".")
    plt.plot(x, coefs[1]*x, color='cyan',
         linewidth=3) 
    plt.ylabel('GSRT')
    plt.xlabel('Vertex X Value')
    plt.title(region)              
    plt.legend(handles=[cyan_line,red_line],loc=9,prop={'size':12})
    plt.savefig(''.join(['/home/bschloss/pul8_read/MRI/Data/Length_Vertex/',region,'_vertex_gsrt_avgwl_regression.png']))
    plt.clf()
    
