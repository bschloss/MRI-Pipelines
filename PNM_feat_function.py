# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np

fsf = open('/gpfs/group/pul8/default/read/Scripts/PNM_ACI.fsf','r').read()
variables = [word.rstrip('"').lstrip('"') for word in fsf.split() if '$' in word]
variables.reverse()
pardir = '/Users/Tanner_Quiggle/Desktop/fMRI_LAB/Resting_State/002/'

for var in variables:
    if var == '$OUTPUTDIR':
        fsf = fsf.replace(var,''.join([pardir,'PNM_Test']))
    if var == '$RESTDATA':
        rd = pardir + 'Rest_Preprocessing/Coregistered_F2S/_Register_F2S0/'
        rd += os.listdir(rd)[0]
        fsf = fsf.replace(var,rd)
    if '$EV' == var[:3]:
        evnum = var[3:]
        evdig = int(evnum) - 2
        if not (int(evnum) == 1 or int(evnum) == 2):
            evf = pardir + 'Rest_Preprocessing/PNM_EVs/PNMev'
            if evdig<10:
                evf += '00'
            elif evdig<100:
                evf += '0'
            evf += str(evdig) + '.nii.gz'
            print var,evf
            fsf = fsf.replace(var,evf)
        else:
            if int(evnum) == 1:
                wmts = ''.join([pardir,'Rest_Preprocessing/WM_TS/_WM_Time_Series_Averager0/epi2struct_ts.txt'])
                wmts = np.asarray([float(num.rstrip('\n')) for num in open(wmts,'r').readlines()])
                wmts = wmts - np.mean(wmts)
                np.savetxt('/tmp/wmts.run001.txt',wmts)
                fsf = fsf.replace(var,'/tmp/wmts.run001.txt')
            if int(evnum) == 2:
                csfts = ''.join([pardir,'Rest_Preprocessing/CS_TS/_CS_Time_Series_Averager0/epi2struct_ts.txt'])
                csfts = np.asarray([float(num.rstrip('\n')) for num in open(csfts,'r').readlines()])
                csfts = csfts - np.mean(wmts)
                np.savetxt('/tmp/csfts.run001.txt',csfts)
                fsf = fsf.replace(var,'/tmp/csfts.run001.txt')
                
open(''.join([pardir,'fsf002.fsf']),'w').write(fsf)
