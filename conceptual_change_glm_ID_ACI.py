# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 03:39:49 2017

@author: bschloss
"""

import os
import shutil
import string
import exceptions
import numpy.random as random
import nibabel as nib
import numpy as np
import nipype.interfaces.io as nio
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface as ID
import nipype.interfaces.fsl as fsl
import argparse as ap
import subprocess as sp

def nest_list(in_file):
    if len(in_file) == 1:
        out = in_file
    else:
        out = [in_file]
    print in_file
    print out
    return out
    
def unnest_list(in_file):
    if len(in_file[0]) == 1:
	out = in_file
    else:
	out = in_file[0]
    print in_file
    print out
    return out
        
#Define the directories where data is located
parser = ap.ArgumentParser(description='Preprocess DTI data and put in new folder')
parser.add_argument('pardir', metavar="stanford", type=str,
                    help="Path to participant's directory")                                                  
args = parser.parse_args()
pardir= '/gpfs/group/pul8/default/read/'
if len(args.pardir) == 1:
    pardir += '00' + args.pardir + '/'
elif len(args.pardir) == 2:
    pardir += '0' + args.pardir + '/'
else:
    pardir += args.pardir + '/'  

#Collect and sort event files for model
target = ['axial',
          'battery',
          'boiler',
          'building',
          'canyon',
          'cheese',
          'circle',
          'circuit',
          'coast',
          'combination',
          'current',
          'distance',
          'dust',
          'earth',
          'electric',
          'electron',
          'engineer',
          'gps',
          'gravity',
          'ground',
          'hull',
          'human',
          'instrument',
          'item',
          'lightbulb',
          'live',
          'location',
          'loop',
          'mars',
          'number',
          'ocean',
          'oil',
          'orbit',
          'order',
          'pasta',
          'path',
          'pepperoni',
          'permutation',
          'piece',
          'pizza',
          'plant',
          'propeller',
          'radio',
          'receiver',
          'rice',
          'rover',
          'safety',
          'satellite',
          'sausage',
          'scientist',
          'select',
          'set',
          'ship',
          'shore',
          'signal',
          'source',
          'space',
          'spacecraft',
          'spill',
          'station',
          'storm',
          'sun',
          'temperature',
          'tilt',
          'time',
          'wave',
          'wire',
          'world']
target = ['_'.join([word,'target']) for word in target] 
ev_dir = pardir + 'EV2/'
FSFDIR = pardir + 'FSF/'
try:
    os.mkdir(FSFDIR)
except:
    exceptions.OSError
FSFDIR = FSFDIR + 'IFID/'
try:
    os.mkdir(FSFDIR)
except:
    exceptions.OSError
outputdir = pardir + 'fMRI_Analyses/Conceptual_Change_ID/'
try:
    os.mkdir(outputdir)
except:
    exceptions.OSError
tr = .400000000
ndel = 0
dwell = .58
te = 30
standard = ''
hp_cutoff = 128

for word in target: 
    runnum = 0
    for i in range(1,6):
        ev_file = pardir + 'EV2/Run' + str(i) + '/' + word + '1.run00' + str(i) + '.txt'
        #Only include runs where the participants fixated on that word
        if os.path.isfile(ev_file):
            wordnum = 1
            FSFDIR = pardir + 'FSF/IFID/' + word.capitalize() + str(wordnum) + '/'
            outputdir = pardir + 'fMRI_Analyses/Conceptual_Change_ID/' + word.capitalize() + str(wordnum) + '/'
            while(os.path.isdir(FSFDIR)):
                wordnum += 1
                FSFDIR = pardir + 'FSF/IFID/' + word.capitalize() + str(wordnum) + '/'
                outputdir = pardir + 'fMRI_Analyses/Conceptual_Change_ID/' + word.capitalize() + str(wordnum) + '/'
            try:
                os.mkdir(FSFDIR)
            except:
                exceptions.OSError
            #Create a temporary directory
            out_base = ''
            while out_base == '':
                letters = string.ascii_letters
                out_base = ''.join(['/tmp/tmp',''.join([letters[j] for j in random.choice(len(letters),10)])])
                out_base += '/Rename_EVs/'
                try:
                    os.makedirs(out_base)
                except:
                    exceptions.OSError
                    out_base = ''       
            
            if not (pardir[-5:-1]=='/021' and i == 4) and not (pardir[-5:-1] in ['/014','2006','2014'] and i == 5) or (pardir[-5:-1] == '2005' and i ==2):
                if pardir[-5:-1]=='/021' and i > 4:
                    adjustment = 2
                elif pardir[-5:-1] == '2005' and i > 2:
                    adjustment = 2
                else: 
                    adjustment = 1
                csf_ts = pardir + 'fMRI_Preprocessing/Trimmed_CSF/_CSF_Trimmer' + str(i-adjustment) + '/' 
                csf_ts += os.listdir(csf_ts)[0] 
                wm_ts = pardir + 'fMRI_Preprocessing/Trimmed_WM/_WM_Trimmer' + str(i-adjustment) + '/' 
                wm_ts += os.listdir(wm_ts)[0] 
                ven_ts = pardir + 'fMRI_Preprocessing/Trimmed_Ven/_Ven_Trimmer' + str(i-adjustment) + '/' 
                ven_ts += os.listdir(ven_ts)[0] 
                ev_physio = [wm_ts,csf_ts,ven_ts]
                runnum += 1
                os.makedirs(''.join([out_base,'Run1/']))
                sd = ev_dir + 'Run' + str(i) + '/'
                dd = out_base + 'Run1/'
                #Rename EVs to have the correct number of runs and store in temporary direcotry
                ev_target = []
                for l in ['_'.join([word.replace('_target',''),'Other_Content_or_Target']),'No_Interest']:
                    f_src = sd + l + '.run00' + str(i) + '.txt'
                    f_dest = dd + l + '.run001.txt'
                    shutil.copy(f_src,f_dest)
                    ev_target.append(f_dest)
                fix_num = 1
                while (os.path.isfile(''.join([sd,word,str(fix_num),'.run00',str(i),'.txt']))):
                    f_src = sd + word + str(fix_num) + '.run00' + str(i) + '.txt'
                    f_dest = dd + word + str(fix_num) + '.run001.txt'
                    shutil.copy(f_src,f_dest)
                    ev_target.append(f_dest)
                    fix_num += 1
                trim_vol_file = sd + 'text_end_vol_num.run00' + str(i) + '.txt'
                func_trim_point = int(open(trim_vol_file,'r').read())
                func = ''.join([pardir,'fMRI_Preprocessing/Trimmed_Func_Highpass/_Question_Trimmer_Highpass',str(i-1),'/'])
                func += os.listdir(func)[0]
                mp = ''.join([pardir,'fMRI_Preprocessing/Trimmed_MotionPars/_Motion_Parameter_Trimmer',str(i-1),'/'])
                mp += os.listdir(mp)[0]
                
            numev = len(ev_target)*2 + len(ev_physio)
            numcon = len(ev_target)*4 + len(ev_physio)
            numf = 0
            parfsf = ''.join(open('/gpfs/group/pul8/default/read/Scripts/ID.fsf', 'r').readlines()[:276])
            parfsf = parfsf.replace('OUTPUTDIR',outputdir)            
            parfsf = parfsf.replace('NUMVOLS', str(nib.load(func).get_data().shape[-1]))
            parfsf = parfsf.replace('NDELETE',str(ndel))
            parfsf = parfsf.replace('DWELLTIME',str(dwell))
            parfsf = parfsf.replace('NUMEV',str(numev))
            parfsf = parfsf.replace('NUMCON',str(numcon))
            parfsf = parfsf.replace('NUMF',str(numf))
            parfsf = parfsf.replace('STANDARD',standard)
            parfsf = parfsf.replace('NUMVOX',str(np.prod(nib.load(func).get_data().shape[:3])))
            parfsf = parfsf.replace('HP_CUTOFF',str(hp_cutoff))
            parfsf = parfsf.replace('PARNUMBER', str(pardir[-4:-1]))
            parfsf = parfsf.replace('FUNC',func)
            parfsf = parfsf.replace('MOTIONPARAMETERS',mp)
            parfsf = parfsf.replace('RepetitionTime',str(tr))
            parfsf = parfsf.replace('EchoTime',str(te))
            
            id_temp = ''
            id_temp += '# Basic waveform shape (EV 1)\n'
            id_temp += '# 0 : Square\n'
            id_temp += '# 1 : Sinusoid\n'
            id_temp += '# 2 : Custom (1 entry per volume)\n'
            id_temp += '# 3 : Custom (3 column format)\n'
            id_temp += '# 4 : Interaction\n'
            id_temp += '# 10 : Empty (all zeros)\n'
            id_temp += 'set fmri(shape1) 3\n'
            id_temp += '\n'
            id_temp += '# Convolution (EV 1)\n'
            id_temp += '# 0 : None\n'
            id_temp += '# 1 : Gaussian\n'
            id_temp += '# 2 : Gamma\n'
            id_temp += '# 3 : Double-Gamma HRF\n'
            id_temp += '# 4 : Gamma basis functions\n'
            id_temp += '# 5 : Sine basis functions\n'
            id_temp += '# 6 : FIR basis functions\n'
            id_temp += 'set fmri(convolve1) 2\n'
            id_temp += '\n'
            id_temp += '# Convolve phase (EV 1)\n'
            id_temp += 'set fmri(convolve_phase1) 0\n'
            id_temp += '\n'
            id_temp += '# Apply temporal filtering (EV 1)\n'
            id_temp += 'set fmri(tempfilt_yn1) 1\n'
            id_temp += '\n'
            id_temp += '# Add temporal derivative (EV 1)\n'
            id_temp += 'set fmri(deriv_yn1) 0\n'
            
            dg_temp = ''
            dg_temp += '# Basic waveform shape (EV 2)\n'
            dg_temp += '# 0 : Square\n'
            dg_temp += '# 1 : Sinusoid\n'
            dg_temp += '# 2 : Custom (1 entry per volume)\n'
            dg_temp += '# 3 : Custom (3 column format)\n'
            dg_temp += '# 4 : Interaction\n'
            dg_temp += '# 10 : Empty (all zeros)\n'
            dg_temp += 'set fmri(shape2) 3\n'
            dg_temp += '\n'
            dg_temp += '# Convolution (EV 2)\n'
            dg_temp += '# 0 : None\n'
            dg_temp += '# 1 : Gaussian\n'
            dg_temp += '# 2 : Gamma\n'
            dg_temp += '# 3 : Double-Gamma HRF\n'
            dg_temp += '# 4 : Gamma basis functions\n'
            dg_temp += '# 5 : Sine basis functions\n'
            dg_temp += '# 6 : FIR basis functions\n'
            dg_temp += 'set fmri(convolve2) 3\n'
            dg_temp += '\n'
            dg_temp += '# Convolve phase (EV 2)\n'
            dg_temp += 'set fmri(convolve_phase2) 0\n'
            dg_temp += '\n'  
            dg_temp += '# Apply temporal filtering (EV 2)\n'
            dg_temp += 'set fmri(tempfilt_yn2) 1\n'
            dg_temp += '\n'   
            dg_temp += '# Add temporal derivative (EV 2)\n'
            dg_temp += 'set fmri(deriv_yn2) 0\n'
            
            no_conv_temp = ''
            no_conv_temp += '# Basic waveform shape (EV 1)\n'
            no_conv_temp += '# 0 : Square\n'
            no_conv_temp += '# 1 : Sinusoid\n'
            no_conv_temp += '# 2 : Custom (1 entry per volume)\n'
            no_conv_temp += '# 3 : Custom (3 column format)\n'
            no_conv_temp += '# 4 : Interaction\n'
            no_conv_temp += '# 10 : Empty (all zeros)\n'
            no_conv_temp += 'set fmri(shape1) 2\n'
            no_conv_temp += '\n'
            no_conv_temp += '# Convolution (EV 1)\n'
            no_conv_temp += '# 0 : None\n'
            no_conv_temp += '# 1 : Gaussian\n'
            no_conv_temp += '# 2 : Gamma\n'
            no_conv_temp += '# 3 : Double-Gamma HRF\n'
            no_conv_temp += '# 4 : Gamma basis functions\n'
            no_conv_temp += '# 5 : Sine basis functions\n'
            no_conv_temp += '# 6 : FIR basis functions\n'
            no_conv_temp += 'set fmri(convolve1) 0\n'
            no_conv_temp += '\n'
            no_conv_temp += '# Convolve phase (EV 1)\n'
            no_conv_temp += 'set fmri(convolve_phase1) 0\n'
            no_conv_temp += '\n'
            no_conv_temp += '# Apply temporal filtering (EV 1)\n'
            no_conv_temp += 'set fmri(tempfilt_yn1) 1\n'
            no_conv_temp += '\n'
            no_conv_temp += '# Add temporal derivative (EV 1)\n'
            no_conv_temp += 'set fmri(deriv_yn1) 0\n'
            
            evnum = 0
            for ev in ev_physio:
                evnum += 1
                parfsf += '# EV ' + str(evnum) + ' title\n'
                parfsf += 'set fmri(evtitle' + str(evnum) + ') "' + ev.split('/')[-1][:-11] + '_ID"\n'
                parfsf += no_conv_temp.replace('1)',''.join([str(evnum),')']))
                parfsf += '\n'
                parfsf += '# Custom EV file (EV ' + str(evnum) + ')\n'
                parfsf += 'set fmri(custom' + str(evnum) + ') "' + ev + '"\n'
                parfsf += '\n'
                for j in range(numev + 1):
                    parfsf += '# Orthogonalise EV ' + str(evnum) + ' wrt EV ' + str(j) + '\n'
                    parfsf += 'set fmri(ortho' + str(evnum) + '.' + str(j) + ') 0\n'
                parfsf += '\n'
            for ev in ev_target:
                evnum += 1
                parfsf += '# EV ' + str(evnum) + ' title\n'
                parfsf += 'set fmri(evtitle' + str(evnum) + ') "' + ev.split('/')[-1][:-11] + '_ID"\n'
                parfsf += id_temp.replace('1)',''.join([str(evnum),')']))
                parfsf += '\n'
                parfsf += '# Custom EV file (EV ' + str(evnum) + ')\n'
                parfsf += 'set fmri(custom' + str(evnum) + ') "' + ev + '"\n'
                parfsf += '\n'
                parfsf += '# Gamma sigma (EV ' + str(evnum) + ')\n'
                parfsf += 'set fmri(gammasigma' + str(evnum) + ') .5\n'
                parfsf += '\n'
                parfsf += '# Gamma delay (EV ' + str(evnum) + ')\n'
                parfsf += 'set fmri(gammadelay' + str(evnum) + ') 2\n'
                parfsf += '\n'
                for j in range(numev + 1):
                    parfsf += '# Orthogonalise EV ' + str(evnum) + ' wrt EV ' + str(j) + '\n'
                    parfsf += 'set fmri(ortho' + str(evnum) + '.' + str(j) + ') 0\n'
                parfsf += '\n'
                evnum += 1
                parfsf += '# EV ' + str(evnum) + ' title\n'
                parfsf += 'set fmri(evtitle' + str(evnum) + ') "' + ev.split('/')[-1][:-11] + '_DG"\n'
                parfsf += dg_temp.replace('2)',''.join([str(evnum),')']))
                parfsf += '\n'
                parfsf += '# Custom EV file (EV ' + str(evnum) + ')\n'
                parfsf += 'set fmri(custom' + str(evnum) + ') "' + ev + '"\n'
                parfsf += '\n'
                for j in range(numev + 1):
                    parfsf += '# Orthogonalise EV ' + str(evnum) + ' wrt EV ' + str(j) + '\n'
                    parfsf += 'set fmri(ortho' + str(evnum) + '.' + str(j) + ') 0\n'
                parfsf += '\n'
            parfsf += '# Contrast & F-tests mode\n'
            parfsf += '# real : control real EVs\n'
            parfsf += '# orig : control original EVs\n'
            parfsf += 'set fmri(con_mode_old) orig\n'
            parfsf += 'set fmri(con_mode) orig\n'
            parfsf += '\n'
            
            evnum = 0
            connum = 0
            for ev in ev_physio:
                evnum += 1
                connum += 1
                parfsf += '# Display images for contrast_real ' + str(connum) + '\n'
                parfsf += 'set fmri(conpic_real.' + str(connum) + ') ' + str(connum) + '\n'
                parfsf += '\n'
                parfsf += '# Title for contrast_real ' + str(connum) + '\n'
                parfsf += 'set fmri(conname_real.' + str(connum) + ') "' + ev.split('/')[-1][:-11] + '_No_Conv"\n'
                parfsf += '\n'
                for j in range(1,numev + 1):
                    parfsf += '# Real contrast_real vector ' + str(connum) + ' element ' + str(j) + '\n'
                    if evnum == j:
                        parfsf += 'set fmri(con_real' + str(connum) + '.' + str(j) + ') 1.0\n\n'
                    else:
                        parfsf += 'set fmri(con_real' + str(connum) + '.' + str(j) + ') 0\n\n'
            for ev in ev_target:
                evnum += 1
                connum += 1
                parfsf += '# Display images for contrast_real ' + str(connum) + '\n'
                parfsf += 'set fmri(conpic_real.' + str(connum) + ') ' + str(connum) + '\n'
                parfsf += '\n'
                parfsf += '# Title for contrast_real ' + str(connum) + '\n'
                parfsf += 'set fmri(conname_real.' + str(connum) + ') "' + ev.split('/')[-1][:-11] + '_ID"\n'
                parfsf += '\n'
                for j in range(1,numev + 1):
                    parfsf += '# Real contrast_real vector ' + str(connum) + ' element ' + str(j) + '\n'
                    if evnum == j:
                        parfsf += 'set fmri(con_real' + str(connum) + '.' + str(j) + ') -1.0\n\n'
                    else:
                        parfsf += 'set fmri(con_real' + str(connum) + '.' + str(j) + ') 0\n\n'
                connum += 1
                parfsf += '# Display images for contrast_real ' + str(connum) + '\n'
                parfsf += 'set fmri(conpic_real.' + str(connum) + ') ' + str(connum) + '\n'
                parfsf += '\n'
                parfsf += '# Title for contrast_real ' + str(connum) + '\n'
                parfsf += 'set fmri(conname_real.' + str(connum) + ') "' + ev.split('/')[-1][:-11] + '_ID_Inv"\n'
                parfsf += '\n'
                for j in range(1,numev + 1):
                    parfsf += '# Real contrast_real vector ' + str(connum) + ' element ' + str(j) + '\n'
                    if evnum == j:
                        parfsf += 'set fmri(con_real' + str(connum) + '.' + str(j) + ') 1.0\n\n'
                    else:
                        parfsf += 'set fmri(con_real' + str(connum) + '.' + str(j) + ') 0\n\n'
                evnum += 1
                connum += 1
                parfsf += '# Display images for contrast_real ' + str(connum) + '\n'
                parfsf += 'set fmri(conpic_real.' + str(connum) + ') ' + str(connum) + '\n'
                parfsf += '\n'
                parfsf += '# Title for contrast_real ' + str(connum) + '\n'
                parfsf += 'set fmri(conname_real.' + str(connum) + ') "' + ev.split('/')[-1][:-11] + '_DG"\n'
                parfsf += '\n'
                for j in range(1,numev + 1):
                    parfsf += '# Real contrast_real vector ' + str(connum) + ' element ' + str(j) + '\n'
                    if evnum == j:
                        parfsf += 'set fmri(con_real' + str(connum) + '.' + str(j) + ') 1.0\n\n'
                    else:
                        parfsf += 'set fmri(con_real' + str(connum) + '.' + str(j) + ') 0\n\n'
                connum += 1
                parfsf += '# Display images for contrast_real ' + str(connum) + '\n'
                parfsf += 'set fmri(conpic_real.' + str(connum) + ') ' + str(connum) + '\n'
                parfsf += '\n'
                parfsf += '# Title for contrast_real ' + str(connum) + '\n'
                parfsf += 'set fmri(conname_real.' + str(connum) + ') "' + ev.split('/')[-1][:-11] + '_DG_Inv"\n'
                parfsf += '\n'
                for j in range(1,numev + 1):
                    parfsf += '# Real contrast_real vector ' + str(connum) + ' element ' + str(j) + '\n'
                    if evnum == j:
                        parfsf += 'set fmri(con_real' + str(connum) + '.' + str(j) + ') -1.0\n\n'
                    else:
                        parfsf += 'set fmri(con_real' + str(connum) + '.' + str(j) + ') 0\n\n'
                
            evnum = 0
            connum = 0
            for ev in ev_physio:
                evnum += 1
                connum += 1
                parfsf += '# Display images for contrast_orig ' + str(connum) + '\n'
                parfsf += 'set fmri(conpic_orig.' + str(connum) + ') ' + str(connum) + '\n'
                parfsf += '\n'
                parfsf += '# Title for contrast_orig ' + str(connum) + '\n'
                parfsf += 'set fmri(conname_orig.' + str(connum) + ') "' + ev.split('/')[-1][:-11] + '_No_Conv"\n'
                parfsf += '\n'
                for j in range(1,numev + 1):
                    parfsf += '# Real contrast_orig vector ' + str(connum) + ' element ' + str(j) + '\n'
                    if evnum == j:
                        parfsf += 'set fmri(con_orig' + str(connum) + '.' + str(j) + ') 1.0\n\n'
                    else:
                        parfsf += 'set fmri(con_orig' + str(connum) + '.' + str(j) + ') 0\n\n'
            for ev in ev_target:
                evnum += 1
                connum += 1
                parfsf += '# Display images for contrast_orig ' + str(connum) + '\n'
                parfsf += 'set fmri(conpic_orig.' + str(connum) + ') ' + str(connum) + '\n'
                parfsf += '\n'
                parfsf += '# Title for contrast_orig ' + str(connum) + '\n'
                parfsf += 'set fmri(conname_orig.' + str(connum) + ') "' + ev.split('/')[-1][:-11] + '_ID"\n'
                parfsf += '\n'
                for j in range(1,numev + 1):
                    parfsf += '# Real contrast_orig vector ' + str(connum) + ' element ' + str(j) + '\n'
                    if evnum == j:
                        parfsf += 'set fmri(con_orig' + str(connum) + '.' + str(j) + ') -1.0\n\n'
                    else:
                        parfsf += 'set fmri(con_orig' + str(connum) + '.' + str(j) + ') 0\n\n'
                connum += 1
                parfsf += '# Display images for contrast_orig ' + str(connum) + '\n'
                parfsf += 'set fmri(conpic_orig.' + str(connum) + ') ' + str(connum) + '\n'
                parfsf += '\n'
                parfsf += '# Title for contrast_orig ' + str(connum) + '\n'
                parfsf += 'set fmri(conname_orig.' + str(connum) + ') "' + ev.split('/')[-1][:-11] + '_ID_Inv"\n'
                parfsf += '\n'
                for j in range(1,numev + 1):
                    parfsf += '# Real contrast_orig vector ' + str(connum) + ' element ' + str(j) + '\n'
                    if evnum == j:
                        parfsf += 'set fmri(con_orig' + str(connum) + '.' + str(j) + ') 1.0\n\n'
                    else:
                        parfsf += 'set fmri(con_orig' + str(connum) + '.' + str(j) + ') 0\n\n'
                evnum += 1
                connum += 1
                parfsf += '# Display images for contrast_orig ' + str(connum) + '\n'
                parfsf += 'set fmri(conpic_orig.' + str(connum) + ') ' + str(connum) + '\n'
                parfsf += '\n'
                parfsf += '# Title for contrast_orig ' + str(connum) + '\n'
                parfsf += 'set fmri(conname_orig.' + str(connum) + ') "' + ev.split('/')[-1][:-11] + '_DG"\n'
                parfsf += '\n'
                for j in range(1,numev + 1):
                    parfsf += '# Real contrast_orig vector ' + str(connum) + ' element ' + str(j) + '\n'
                    if evnum == j:
                        parfsf += 'set fmri(con_orig' + str(connum) + '.' + str(j) + ') 1.0\n\n'
                    else:
                        parfsf += 'set fmri(con_orig' + str(connum) + '.' + str(j) + ') 0\n\n'    
                connum += 1
                parfsf += '# Display images for contrast_orig ' + str(connum) + '\n'
                parfsf += 'set fmri(conpic_orig.' + str(connum) + ') ' + str(connum) + '\n'
                parfsf += '\n'
                parfsf += '# Title for contrast_orig ' + str(connum) + '\n'
                parfsf += 'set fmri(conname_orig.' + str(connum) + ') "' + ev.split('/')[-1][:-11] + '_DG_Inv"\n'
                parfsf += '\n'
                for j in range(1,numev + 1):
                    parfsf += '# Real contrast_orig vector ' + str(connum) + ' element ' + str(j) + '\n'
                    if evnum == j:
                        parfsf += 'set fmri(con_orig' + str(connum) + '.' + str(j) + ') -1.0\n\n'
                    else:
                        parfsf += 'set fmri(con_orig' + str(connum) + '.' + str(j) + ') 0\n\n'  
            
            parfsf += '# Contrast masking - use >0 instead of thresholding?\n'
            parfsf += 'set fmri(conmask_zerothresh_yn) 0\n\n'
            evnum = 0
            for j in range(1,numcon + 1):
                for k in range(1,numcon + 1):
                    if j != k:
                        parfsf += '# Mask real contrast/F-test ' + str(j) + ' with real contrast/F-test ' + str(k) + '?\n'
                        parfsf += 'set fmri(conmask' + str(j) + '_' + str(k) + ') 0\n\n'
            parfsf += '# Do contrast masking at all?\n'
            parfsf += 'set fmri(conmask1_1) 0\n'
            parfsf += '\n'
            parfsf += '##########################################################\n'
            parfsf += "# Now options that don't appear in the GUI\n"
            parfsf += '\n'
            parfsf += '# Alternative (to BETting) mask image\n'
            parfsf += 'set fmri(alternative_mask) ""\n'
            parfsf += '\n'
            parfsf += '# Initial structural space registration initialisation transform\n'
            parfsf += 'set fmri(init_initial_highres) ""\n'
            parfsf += '\n'
            parfsf += '# Structural space registration initialisation transform\n'
            parfsf += 'set fmri(init_highres) ""\n'
            parfsf += '\n'
            parfsf += '# Standard space registration initialisation transform\n'
            parfsf += 'set fmri(init_standard) ""\n'
            parfsf += '\n'
            parfsf += '# For full FEAT analysis: overwrite existing .feat output dir?\n'
            parfsf += 'set fmri(overwrite_yn) 0\n'
            
            open("".join([FSFDIR,'design.fsf']), 'w').write(parfsf)
            parfsf = "".join([FSFDIR,'design.fsf'])
            os.chdir(FSFDIR)
            model = sp.Popen(['feat_model','design'])
            model.wait()
            design_mat = FSFDIR + 'design.mat'
            design_png = FSFDIR + 'design.png'
            design_con = FSFDIR + 'design.con'
            if mp and not os.path.isdir(''.join([pardir,'fMRI_Analyses/Conceptual_Change_ID/',word.capitalize().replace('_target',''),str(runnum)])):
                #Run analysis                      
                analysis = Workflow(name = "Target_GLM_Level1_ID")
                data = Node(ID(fields=['func','parfsf','design_mat',
                                       'design_png','con_file'],
                               mandatory_inputs=False),name='Data')
                data.inputs.func = func
                data.inputs.parfsf = parfsf
                data.inputs.design_mat = design_mat
                data.inputs.design_png = design_png
                data.inputs.con_file = design_con
                
                #Use the DataSink function to store all outputs
                datasink = Node(nio.DataSink(), name= 'Output')
                datasink.inputs.base_directory = pardir + 'fMRI_Analyses/Conceptual_Change_ID/' + word.capitalize().replace('_target','') + str(runnum)
                
                #Output the fsf file
                analysis.connect(data,'parfsf',datasink,'FSF')
                
                #Output the design matrix
                analysis.connect(data,'design_mat',datasink,'Design_Matrix')
                
                #Output an image of the design matrix
                analysis.connect(data,'design_png',datasink,'Design_Matrix_Img')
                
                #Now we will run the model on each voxel in the brain and output first level stats maps
                filmgls = MapNode(interface=fsl.model.FILMGLS(threshold=0.00,smooth_autocorr=True),
                             name = "Parametric_Fit_Model",
                             iterfield = ['in_file','design_file','tcon_file'])
                analysis.connect(data,'func',filmgls,'in_file')
                analysis.connect(data,'design_mat',filmgls,'design_file')
                analysis.connect(data, 'con_file',filmgls,'tcon_file')
                analysis.connect(filmgls,'copes',datasink,'Parametric_Copes')
                analysis.connect(filmgls,'dof_file',datasink,'Parametric_DOF')
                analysis.connect(filmgls,'fstats',datasink,'Parametric_Fstats')
                analysis.connect(filmgls,'param_estimates',datasink,'Parametric_Param_Estimates')
                analysis.connect(filmgls,'residual4d',datasink,'Parametric_Residual4D')
                analysis.connect(filmgls,'sigmasquareds',datasink,'Parametric_Sigma_Squareds')
                analysis.connect(filmgls,'thresholdac',datasink,'Parametric_AC_Params')
                analysis.connect(filmgls,'tstats',datasink,'Parametric_Tstats')
                analysis.connect(filmgls,'varcopes',datasink,'Parametric_Varcopes')
                analysis.connect(filmgls,'zfstats',datasink,'Parametric_Z_Fstats')
                analysis.connect(filmgls,'zstats',datasink,'Parametric_Z_Tstats')
                
                analysis.run()
