# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:11:57 2017

@author: bschloss
"""
import os
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface as ID
from nipype.interfaces.utility import Function
import argparse as ap

def motionpar_trimmer(mp,ftp):
    import os
    mp_trimmed = open(mp,'r').readlines()[:ftp]
    trimmed = ''
    for line in mp_trimmed:
		trimmed += line
    trimmed = trimmed.rstrip('\n')
    file_name = os.getcwd() + '/' + mp.split('/')[-1].replace('.par','_trimmed.par').replace('.nii.gz','')
    with open(file_name,'w') as f:
        f.write(trimmed)
    return file_name

def physio_trimmer(physio,ftp):
    import os
    trimmed = ''.join(open(physio,'r').readlines()[:ftp]).rstrip(' \n')
    file_name = os.getcwd() + '/' + 'physio_ts_trimmed.run00' + str(int(os.getcwd()[-1]) + 1) + '.txt'
    with open(file_name,'w') as f:
        f.write(trimmed)
    return file_name
    
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
    
funcsmoothed = pardir + 'fMRI_Preprocessing/Smoothed/'
funcsmoothed = [funcsmoothed + '_Smoother' + str(i) + '/' for i in range(5)]
funcsmoothed = [d + os.listdir(d)[0] for d in funcsmoothed]

funchighpass = pardir + 'fMRI_Preprocessing/Smoothed/'
funchighpass = [funchighpass + '_Smoother' + str(i) + '/' for i in range(5)]
funchighpass = [d + os.listdir(d)[0] for d in funchighpass]

mp = pardir + 'fMRI_Preprocessing/MotionPars/'
mp = [mp + '_realign' + str(i) + '/' for i in range(5)] 
mp = [d + os.listdir(d)[0] for d in mp]     

csf_ts = [pardir + 'fMRI_Preprocessing/CSF_TS/_CSF_Time_Series_Averager' + str(i) + '/' for i in range(5)]
csf_ts = [d + os.listdir(d)[0] for d in csf_ts if os.path.isdir(d)]
wm_ts = [pardir + 'fMRI_Preprocessing/WM_TS/_WM_Time_Series_Averager' + str(i) + '/' for i in range(5)]
wm_ts = [d + os.listdir(d)[0] for d in wm_ts if os.path.isdir(d)]
ven_ts = [pardir + 'fMRI_Preprocessing/Avg_Ventricle_TS/_Vent_Time_Series_Averager' + str(i) + '/' for i in range(5)]
ven_ts = [d + os.listdir(d)[0] for d in ven_ts if os.path.isdir(d)]

print ven_ts
            
ev_dir = pardir + 'EV/'
if pardir.rstrip('/')[-4:] == '/014' or pardir.rstrip('/')[-4:] == '2006' or pardir.rstrip('/')[-4:] == '2014':
    funchighpass.pop(4)
    funcsmoothed.pop(4)
    mp.pop(4)
    csf_ts.pop(4)
    wm_ts.pop(4)
    ven_ts.pop(4)

if pardir.rstrip('/')[-4:] == '/021':
    funchighpass.pop(3)
    funcsmoothed.pop(3)
    mp.pop(3)
    csf_ts.pop(3)
    wm_ts.pop(3)
    ven_ts.pop(3)

if pardir.rstrip('/')[-4:] == '2005':
    funchighpass.pop(1)
    funcsmoothed.pop(1)
    mp.pop(1)
    csf_ts.pop(1)
    wm_ts.pop(1)
    ven_ts.pop(1)

#Collect and sort event files for model
func_trim_point = []

for i in range(1,6):
    if not (pardir.rstrip('/')[-4:] == '/014' and i==5) and not (pardir.rstrip('/')[-4:] == '/021' and i==4) and not (pardir.rstrip('/')[-4:] in ['2006','2014'] and i==5)  and not (pardir.rstrip('/')[-4:] == '2005' and i == 2):
    	d = ev_dir + 'Run' + str(i) + '/'
    	trim_vol_file = d + 'text_end_vol_num.run00' + str(i) + '.txt'
    	func_trim_point.append(int(open(trim_vol_file,'r').read()))
    
analysis = Workflow(name = "fMRI_Analyses_Low_Level")
data = Node(ID(fields=['funcsmoothed','funchighpass','mp',
                       'wm_ts','csf_ts','ven_ts','func_trim_point'],
               mandatory_inputs=False),name='Data')
data.inputs.funcsmoothed = funcsmoothed
data.inputs.funchighpass = funchighpass
data.inputs.mp = mp
data.inputs.wm_ts = wm_ts
data.inputs.csf_ts = csf_ts
data.inputs.ven_ts = ven_ts
data.inputs.func_trim_point = func_trim_point
data.inputs.mask = '/home/bschloss/pul8_read/MRI/Data/MNI152_T1_3mm3mm4mm_brain_mask.nii.gz'

#Use the DataSink function to store all outputs
datasink = Node(nio.DataSink(), name= 'Output')
datasink.inputs.base_directory = pardir + 'fMRI_Preprocessing/'
'''
question_trimmer_smoothed = MapNode(fsl.ExtractROI(t_min=0),
                      name='Question_Trimmer_Smoothed',
                      iterfield=['in_file','t_size'])
                      
analysis.connect(data,'funcsmoothed',question_trimmer_smoothed,'in_file')
analysis.connect(data,'func_trim_point',question_trimmer_smoothed,'t_size')
analysis.connect(question_trimmer_smoothed,'roi_file',datasink,'Trimmed_Func_Smoothed')

question_trimmer_highpass = MapNode(fsl.ExtractROI(t_min=0),
                      name='Question_Trimmer_Highpass',
                      iterfield=['in_file','t_size'])
                      
analysis.connect(data,'funchighpass',question_trimmer_highpass,'in_file')
analysis.connect(data,'func_trim_point',question_trimmer_highpass,'t_size')
analysis.connect(question_trimmer_highpass,'roi_file',datasink,'Trimmed_Func_Highpass')

mp_trimmer = MapNode(Function(input_names=['mp','ftp'],
                                    output_names=['trimmed'],
                                    function=motionpar_trimmer),
                                    name='Motion_Parameter_Trimmer',
                                    iterfield=['mp','ftp'])
analysis.connect(data,'mp',mp_trimmer,'mp')
analysis.connect(data,'func_trim_point',mp_trimmer,'ftp')
analysis.connect(mp_trimmer,'trimmed',datasink,'Trimmed_MotionPars')
'''
wm_trimmer = MapNode(Function(input_names=['physio','ftp'],
                                    output_names=['trimmed'],
                                    function=physio_trimmer),
                                    name='WM_Trimmer',
                                    iterfield=['physio','ftp'])
analysis.connect(data,'wm_ts',wm_trimmer,'physio')
analysis.connect(data,'func_trim_point',wm_trimmer,'ftp')
analysis.connect(wm_trimmer,'trimmed',datasink,'Trimmed_WM')

csf_trimmer = MapNode(Function(input_names=['physio','ftp'],
                                    output_names=['trimmed'],
                                    function=physio_trimmer),
                                    name='CSF_Trimmer',
                                    iterfield=['physio','ftp'])
analysis.connect(data,'csf_ts',csf_trimmer,'physio')
analysis.connect(data,'func_trim_point',csf_trimmer,'ftp')
analysis.connect(csf_trimmer,'trimmed',datasink,'Trimmed_CSF')

ven_trimmer = MapNode(Function(input_names=['physio','ftp'],
                                    output_names=['trimmed'],
                                    function=physio_trimmer),
                                    name='Ven_Trimmer',
                                    iterfield=['physio','ftp'])
analysis.connect(data,'ven_ts',ven_trimmer,'physio')
analysis.connect(data,'func_trim_point',ven_trimmer,'ftp')
analysis.connect(ven_trimmer,'trimmed',datasink,'Trimmed_Ven')

analysis.run(plugin='MultiProc', plugin_args={'n_procs' : 5})
