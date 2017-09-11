# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 14:10:35 2016

@author: bschloss
"""
import os
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
from nipype.pipeline.engine import Workflow, Node, MapNode
import nipype.algorithms.modelgen as model
from nipype.interfaces.utility import IdentityInterface as ID
from nipype.interfaces.utility import Function
import argparse as ap
import shutil as sh
import exceptions
import numpy as np
import nibabel as nib
import subprocess as sp
    
def list_of_lists(in_file):
    return [[item] for item in in_file]

def nest_list(in_file):
    return [in_file]
    
def unnest_list(in_file):
    new = []
    for l in in_file:
        new = new + l
    return new
    
def printfiles(in_files):
    print in_files
    return in_files
    
def sort_copes(copes):
    output = [[] for i in range(len(copes[0]))]
    for i in range(len(copes[0])):
        for j in range(len(copes)):
            output[i].append(copes[j][i])
    return output

def family_wise_error(zstat, mask):
    import subprocess as sp
    import os
    import numpy.random as random
    import string
    import exceptions
    import shutil as sh
    out_base = ''
    while out_base == '':
        letters = string.ascii_letters
        out_base = ''.join(['/tmp/tmp',''.join([letters[i] for i in random.choice(len(letters),10)])])
        out_base += '/FWE/'
        try:
            os.makedirs(out_base)
        except:
            exceptions.OSError
            out_base = ''       
    p = sp.Popen(['smoothest', '-z',zstat,'-m',mask], stdout=sp.PIPE, 
                 stderr=sp.PIPE)
    output, error = p.communicate()
    vol = float(output.split('\n')[1].split()[1])
    resels = float(output.split('\n')[2].split()[1])
    p = sp.Popen(['ptoz','.05','-g',"{0:.4f}".format(vol/resels)], stdout=sp.PIPE, 
                 stderr=sp.PIPE)
    output, error = p.communicate()
    thr = float(output.rstrip('\n'))
    sp.call(['fslmaths',zstat,'-thr',"{0:.4f}".format(thr),'-mas',mask,''.join([out_base,zstat.split('/')[-1].replace('.nii.gz','_corrz')])])
    sh.move(os.path.abspath(''.join([out_base,zstat.split('/')[-1].replace('.nii.gz','_corrz.nii.gz')])),'/'.join([os.getcwd(),zstat.split('/')[-1].replace('.nii.gz','_corrz.nii.gz')]))    
    return '/'.join([os.getcwd(),zstat.split('/')[-1].replace('.nii.gz','_corrz.nii.gz')])
    
def false_discovery_rate(zstat,mask):
    import subprocess as sp
    import os
    import numpy.random as random
    import string
    import exceptions
    import shutil as sh
    out_base = ''
    while out_base == '':
        letters = string.ascii_letters
        out_base = ''.join(['/tmp/tmp',''.join([letters[i] for i in random.choice(len(letters),10)])])
        out_base += '/FDR/'
        try:
            os.makedirs(out_base)
        except:
            exceptions.OSError
            out_base = ''       
    sp.call(['fslmaths',zstat,'-ztop',''.join([out_base,zstat.split('/')[-1].replace('.nii.gz','_uncorrp')])])
    pimage = ''.join([out_base,zstat.split('/')[-1].replace('.nii.gz','_uncorrp.nii.gz')])
    p = sp.Popen(['fdr','-i',pimage,'-m',mask,'-q','0.05'], stdout=sp.PIPE, 
                 stderr=sp.PIPE)
    output,error = p.communicate()
    thr = 1.0000 + (-1.0000)*float(output.split('\n')[1])
    p = sp.Popen(['fslmaths',pimage,'-mul','-1','-add','1','-thr',"{0:.4f}".format(thr),'-mas',mask,pimage.replace('_uncorrp.nii.gz','_corr_1minusp')])
    output,error = p.communicate()
    sh.move(os.path.abspath(pimage.replace('_uncorrp.nii.gz','_corr_1minusp.nii.gz')),'/'.join([os.getcwd(),pimage.split('/')[-1].replace('_uncorrp.nii.gz','_corr_1minusp.nii.gz')]))
    return '/'.join([os.getcwd(),pimage.split('/')[-1].replace('_uncorrp.nii.gz','_corr_1minusp.nii.gz')])
            
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

func = pardir + 'fMRI_Preprocessing/Trimmed_Func_Smoothed/'
func = [func + '_Question_Trimmer_Smoothed' + str(i) + '/' for i in range(5)]
func = [d + os.listdir(d)[0] for d in func if os.path.isdir(d)]
mp = pardir + 'fMRI_Preprocessing/Trimmed_MotionPars/'
mp = [mp + '_Motion_Parameter_Trimmer' + str(i) + '/' for i in range(5)] 
mp = [d + os.listdir(d)[0] for d in mp if os.path.isdir(d)]                    

csf_ts = [pardir + 'fMRI_Preprocessing/Trimmed_CSF/_CSF_Trimmer' + str(i) + '/' for i in range(5)]
csf_ts = [d + os.listdir(d)[0] for d in csf_ts if os.path.isdir(d)]
wm_ts = [pardir + 'fMRI_Preprocessing/Trimmed_WM/_WM_Trimmer' + str(i) + '/' for i in range(5)]
wm_ts = [d + os.listdir(d)[0] for d in wm_ts if os.path.isdir(d)]
ven_ts = [pardir + 'fMRI_Preprocessing/Trimmed_Ven/_Ven_Trimmer' + str(i) + '/' for i in range(5)]
ven_ts = [d + os.listdir(d)[0] for d in ven_ts if os.path.isdir(d)]
ev_physio = zip(wm_ts,csf_ts,ven_ts)   
ev_dir = pardir + 'EV3/'

#Collect and sort event files for model
ev_parametric = []
labels = ['Content_or_Target_Word','first_instructions','No_Interest','PC1_Length_Content_or_Target',
                     'PC1_Length_Content_or_Target_Squared','Log_Freq_HAL_Content_or_Target',
                     'Log_Freq_HAL_Content_or_Target_Squared']
                     #'Length',
                     #'Length_Squared','Concreteness','Concreteness_Squared',
                     #'Log_Freq_HAL','Log_Freq_HAL_Squared','OLD','OLD_Squared',
                     #'PLD','PLD_Squared','Position','Position_Squared','AoA',
                     #'AoA_Squared','NPhon','NPhon_Squared','NSyll','NSyll_Squared']

for i in range(1,6):
    d = ev_dir + 'Run' + str(i) + '/'
    run = []
    for l in labels:
        f = d + l + '.run00' + str(i) + '.txt'
        run.append(f)
    ev_parametric.append(run)
                                                             
if pardir.rstrip('/')[-4:] == '2006' or pardir.rstrip('/')[-4:] == '2014':
    if len(func) == 5:
        func.pop(4)
    if len(mp) == 5:
        mp.pop(4)
    if len(ev_parametric) == 5:
        ev_parametric.pop(4)
    if len(ev_physio) == 5:
        ev_physio.pop(4)

if pardir.rstrip('/')[-4:] == '/021':
    func.pop(3)
    mp.pop(3)
    ev_parametric.pop(3)
    new_evs_run5 = [f.replace('run005','run004') for f in ev_parametric[3]]
    for old,new in zip(ev_parametric[3],new_evs_run5):
        sh.copy(old,new)
    ev_parametric[3] = new_evs_run5

if pardir.rstrip('/')[-4:] == '2005':
    func = pardir + 'fMRI_Preprocessing/Trimmed_Func_Smoothed/'
    func = [func + '_Question_Trimmer_Smoothed' + str(i) + '/' for i in [0,2,3,4]]
    func = [d + os.listdir(d)[0] for d in func]
    mp = pardir + 'fMRI_Preprocessing/Trimmed_MotionPars/'
    mp = [mp + '_Motion_Parameter_Trimmer' + str(i) + '/' for i in [0,2,3,4]] 
    mp = [d + os.listdir(d)[0] for d in mp]   
    del ev_parametric[1]
    new_evs_run3 = [f.replace('run003','run002') for f in ev_parametric[1]]
    new_evs_run4 = [f.replace('run004','run003') for f in ev_parametric[2]]
    new_evs_run5 = [f.replace('run005','run004') for f in ev_parametric[3]]
    for old,new in zip(ev_parametric[1],new_evs_run3):
        sh.copy(old,new)
    for old,new in zip(ev_parametric[2],new_evs_run4):
        sh.copy(old,new)
    for old,new in zip(ev_parametric[3],new_evs_run5):
        sh.copy(old,new)
    ev_parametric[1] = new_evs_run3
    ev_parametric[2] = new_evs_run4
    ev_parametric[3] = new_evs_run5
    
FSFDIR = pardir + 'FSF/Parametric/Low_Level_Quad_ID/'
try:
    os.makedirs(FSFDIR)
except:
    exceptions.OSError
    
outputdir = pardir + 'fMRI_Analyses/Low_Level_Quad_ID/'    
runnum = 0
fsfs = []
design_mats = []
design_pngs = []
design_cons = []
for f,m,evs,physio in zip(func,mp,ev_parametric,ev_physio):
    runnum += 1
    FSFRUNDIR = FSFDIR + 'Run' + str(runnum) + '/'
    try:
        os.makedirs(FSFRUNDIR)
    except:
        exceptions.OSError
    tr = .400000000
    ndel = 0
    dwell = .58
    te = 30
    standard = ''
    hp_cutoff = 128
    numev = len(evs)*2 + len(physio)
    numcon = len(evs)*4 + len(physio)
    numf = 0
    parfsf = ''.join(open('/gpfs/group/pul8/default/read/Scripts/ID.fsf', 'r').readlines()[:276])
    parfsf = parfsf.replace('OUTPUTDIR',outputdir)            
    parfsf = parfsf.replace('NUMVOLS', str(nib.load(f).get_data().shape[-1]))
    parfsf = parfsf.replace('NDELETE',str(ndel))
    parfsf = parfsf.replace('DWELLTIME',str(dwell))
    parfsf = parfsf.replace('NUMEV',str(numev))
    parfsf = parfsf.replace('NUMCON',str(numcon))
    parfsf = parfsf.replace('NUMF',str(numf))
    parfsf = parfsf.replace('STANDARD',standard)
    parfsf = parfsf.replace('NUMVOX',str(np.prod(nib.load(f).get_data().shape[:3])))
    parfsf = parfsf.replace('HP_CUTOFF',str(hp_cutoff))
    parfsf = parfsf.replace('PARNUMBER', str(pardir[-4:-1]))
    parfsf = parfsf.replace('FUNC',f)
    parfsf = parfsf.replace('MOTIONPARAMETERS',m)
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
    dg_temp += '# Basic waveform shape (EV 1)\n'
    dg_temp += '# 0 : Square\n'
    dg_temp += '# 1 : Sinusoid\n'
    dg_temp += '# 2 : Custom (1 entry per volume)\n'
    dg_temp += '# 3 : Custom (3 column format)\n'
    dg_temp += '# 4 : Interaction\n'
    dg_temp += '# 10 : Empty (all zeros)\n'
    dg_temp += 'set fmri(shape1) 3\n'
    dg_temp += '\n'
    dg_temp += '# Convolution (EV 1)\n'
    dg_temp += '# 0 : None\n'
    dg_temp += '# 1 : Gaussian\n'
    dg_temp += '# 2 : Gamma\n'
    dg_temp += '# 3 : Double-Gamma HRF\n'
    dg_temp += '# 4 : Gamma basis functions\n'
    dg_temp += '# 5 : Sine basis functions\n'
    dg_temp += '# 6 : FIR basis functions\n'
    dg_temp += 'set fmri(convolve1) 3\n'
    dg_temp += '\n'
    dg_temp += '# Convolve phase (EV 1)\n'
    dg_temp += 'set fmri(convolve_phase1) 0\n'
    dg_temp += '\n'  
    dg_temp += '# Apply temporal filtering (EV 1)\n'
    dg_temp += 'set fmri(tempfilt_yn1) 1\n'
    dg_temp += '\n'   
    dg_temp += '# Add temporal derivative (EV 1)\n'
    dg_temp += 'set fmri(deriv_yn1) 0\n'
    
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
    
    evnum  = 0
    
    for ev in physio:
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
    for ev in evs:
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
        parfsf += dg_temp.replace('1)',''.join([str(evnum),')']))
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
    for ev in physio:
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
    for ev in evs:
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
    for ev in physio:
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
    for ev in evs:
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
    
    open("".join([FSFRUNDIR,'design.fsf']), 'w').write(parfsf)
    parfsf = "".join([FSFRUNDIR,'design.fsf'])
    os.chdir(FSFRUNDIR)
    feat_model = sp.Popen(['feat_model','design',m])
    feat_model.wait()
    design_mat = FSFRUNDIR + 'design.mat'
    design_png = FSFRUNDIR + 'design.png'
    design_con = FSFRUNDIR + 'design.con'
    fsfs.append(parfsf)
    design_mats.append(design_mat)
    design_pngs.append(design_png)
    design_cons.append(design_con)

analysis = Workflow(name = "Low_Level_Quad_ID")
data = Node(ID(fields=['func','fsfs','design_mats',
                       'design_pngs','con_files',
                       'mask'],
               mandatory_inputs=False),name='Data')
data.inputs.func = func
data.inputs.parfsf = fsfs
data.inputs.design_mats = design_mats
data.inputs.design_pngs = design_pngs
data.inputs.con_files = design_cons
data.inputs.mask = '/gpfs/group/pul8/default/read/MNI152_T1_3mm3mm4mm_brain_mask.nii.gz'

#Use the DataSink function to store all outputs
datasink = Node(nio.DataSink(), name= 'Output')
datasink.inputs.base_directory = pardir + 'fMRI_Analyses/Low_Level_Quad_ID'

#Output the fsf file
analysis.connect(data,'fsfs',datasink,'FSF')

#Output the design matrix
analysis.connect(data,'design_mats',datasink,'Design_Matrix')

#Output an image of the design matrix
analysis.connect(data,'design_pngs',datasink,'Design_Matrix_Img')

#Now we will run the model on each voxel in the brain and output first level stats maps
filmgls = MapNode(interface=fsl.model.FILMGLS(threshold=0.00,smooth_autocorr=True),
             name = "Fit_Model",
             iterfield = ['in_file','design_file','tcon_file'])
analysis.connect(data,'func',filmgls,'in_file')
analysis.connect(data,'design_mats',filmgls,'design_file')
analysis.connect(data, 'con_files',filmgls,'tcon_file')
analysis.connect(filmgls,'copes',datasink,'Copes')
analysis.connect(filmgls,'dof_file',datasink,'DOF')
analysis.connect(filmgls,'fstats',datasink,'Fstats')
analysis.connect(filmgls,'param_estimates',datasink,'Param_Estimates')
analysis.connect(filmgls,'residual4d',datasink,'Residual4D')
analysis.connect(filmgls,'sigmasquareds',datasink,'Sigma_Squareds')
analysis.connect(filmgls,'thresholdac',datasink,'AC_Params')
analysis.connect(filmgls,'tstats',datasink,'Tstats')
analysis.connect(filmgls,'varcopes',datasink,'Varcopes')
analysis.connect(filmgls,'zfstats',datasink,'Z_Fstats')
analysis.connect(filmgls,'zstats',datasink,'Z_Tstats')

level2model = Node(fsl.model.L2Model(num_copes = len(mp)),
                                 name = 'L2Model')
#analysis.connect(data,'numcopes',level2model,'num_copes')
analysis.connect(level2model,'design_con',datasink,'L2_TCons')
analysis.connect(level2model,'design_mat',datasink,'L2_Design')
analysis.connect(level2model,'design_grp',datasink,'L2_Group')


copemerge = MapNode(interface=fsl.Merge(dimension='t'),
                          iterfield=['in_files'],
                          name="Cope_Merge")
analysis.connect(filmgls,('copes',sort_copes),copemerge,'in_files')
analysis.connect(copemerge,'merged_file',datasink,'Merged_Copes')

varcopemerge = MapNode(interface=fsl.Merge(dimension='t'),
                       iterfield=['in_files'],
                       name="Varcope_Merge")
analysis.connect(filmgls,('varcopes',sort_copes),varcopemerge,'in_files')
analysis.connect(varcopemerge,'merged_file',datasink,'Merged_Varcopes')
                
flameo = MapNode(interface=fsl.model.FLAMEO(run_mode = 'fe'),
                                                       name = 'Fixed_Effects',
                                                       iterfield = ['cope_file','var_cope_file'])#],'dof_var_cope_file'])  
analysis.connect(copemerge,'merged_file',flameo,'cope_file')
analysis.connect(varcopemerge,'merged_file',flameo,'var_cope_file')
analysis.connect(level2model,'design_con',flameo,'t_con_file')
analysis.connect(level2model,'design_mat',flameo,'design_file')
analysis.connect(level2model,'design_grp',flameo,'cov_split_file')
analysis.connect(data,'mask',flameo,'mask_file')
#analysis.connect(filmgls,'dof_file',flameo,'dof_var_cope_file')
analysis.connect(flameo,'copes',datasink,'Fixed_Effects_Copes')
analysis.connect(flameo,'res4d',datasink,'Fixed_Effects_Residuals')
analysis.connect(flameo,'tdof',datasink,'Fixed_Effects_TDOF')
analysis.connect(flameo,'tstats',datasink,'Fixed_Effects_Tstats')
analysis.connect(flameo,'var_copes',datasink,'Fixed_Effects_Varcopes')
analysis.connect(flameo,'zstats',datasink,'Fixed_Effects_Z_Tstats')

fwe = MapNode(interface=Function(input_names=['zstat','mask'],
                                            output_names=['fwe_corrected'],
                                            function=family_wise_error),
                                    name='FWE',
                                    iterfield=['zstat'])
analysis.connect(flameo,'zstats',fwe,'zstat')
analysis.connect(data,'mask',fwe,'mask')
analysis.connect(fwe,'fwe_corrected',datasink,'FWE')

fdr = MapNode(interface=Function(input_names=['zstat','mask'],
                                            output_names = ['fdr_corrected'],
                                            function=false_discovery_rate),
                                   name='FDR',
                                   iterfield=['zstat'])
analysis.connect(flameo,'zstats',fdr,'zstat')
analysis.connect(data,'mask',fdr,'mask')
analysis.connect(fdr,'fdr_corrected',datasink,'FDR')

smooth_est = MapNode(interface=fsl.SmoothEstimate(),
                                name = "Smooth_Estimate",
                                iterfield = ['zstat_file'])
analysis.connect(flameo,'zstats',smooth_est,'zstat_file')
analysis.connect(data,'mask',smooth_est,'mask_file')

cluster = MapNode(interface=fsl.model.Cluster(minclustersize=True,
                                                         out_localmax_txt_file = True,
                                                         out_localmax_vol_file = True,
                                                         out_index_file=True,
                                                         out_threshold_file = True,
                                                         out_pval_file = True,
                                                         pthreshold = .05),
                                        name = 'Cluster',
                                        iterfield = ['in_file','dlh','volume'],
                                        iterables = ('threshold',[2.33,3.08,3.27,3.62]))
analysis.connect(flameo,'zstats',cluster,'in_file')
analysis.connect(smooth_est,'dlh',cluster,'dlh')
analysis.connect(smooth_est,'volume',cluster,'volume')
analysis.connect(cluster,'index_file',datasink,'Cluster_Index')
analysis.connect(cluster,'localmax_txt_file',datasink,'Cluster_LocalMax_Txt')
analysis.connect(cluster,'localmax_vol_file',datasink,'Cluster_LocalMax_Vol')
analysis.connect(cluster,'threshold_file',datasink,'Cluster_Threshold_File')
analysis.connect(cluster,'pval_file',datasink,'Cluster_Pval_File')
           
analysis.write_graph(dotfilename='fMRI_Preprocessing_Graph.dot',format='svg')
analysis.write_graph(dotfilename='fMRI_Preprocessing_Graph.dot',format='svg',graph2use='exec')
analysis.run(plugin='MultiProc', plugin_args={'n_procs' : 5})