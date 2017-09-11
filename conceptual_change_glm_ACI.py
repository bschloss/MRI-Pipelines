# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 14:10:35 2016

@author: bschloss
"""
import os
import shutil
import string
import exceptions
import numpy.random as random
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
from nipype.pipeline.engine import Workflow, Node, MapNode
import nipype.algorithms.modelgen as model
from nipype.interfaces.utility import IdentityInterface as ID
from nipype.interfaces.utility import Function
import argparse as ap

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
    
def motionpar_trimmer(mp,ftp):
    import os
    import string
    import exceptions
    import numpy.random as random
    mp_trimmed = open(mp,'r').readlines()[:ftp]
    trimmed = ''
    for line in mp_trimmed:
		trimmed += line
    trimmed = trimmed.rstrip('\n')
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
    file_name = out_base + '/' + mp.split('/')[-1].replace('.par','_trimmed.par')
    with open(file_name,'w') as f:
        f.write(trimmed)
    return os.path.abspath(file_name)
            
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
tcon_no_interest_gtbl = ('No_Interest>Baseline','T',['No_Interest'],[1]) 
ev_dir = pardir + 'EV2/'

for word in target:   
    tcon_not_target_fixation_gtbl = ('_'.join([word.replace('_target',''),'Other_Content_or_Target>Baseline']),'T',['_'.join([word.replace('_target',''),'Other_Content_or_Target'])],[1])
    runnum = 0
    for i in range(1,6):
        ev_file = pardir + 'EV2/Run' + str(i) + '/' + word + '1.run00' + str(i) + '.txt'
        func = []
        mp = []
        ev_target = []
        func_trim_point = []
        if os.path.isfile(ev_file):
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
            target_contrasts = [tcon_no_interest_gtbl,tcon_not_target_fixation_gtbl] 
            #Only include runs where the participants fixated on that word
            if not (pardir[-5:-1]=='/021' and i == 4) and not (pardir[-5:-1] == '2005' and i == 2)  and not (pardir[-5:-1] in ['2006','2014'] and i == 5):
                runnum += 1
                os.makedirs(''.join([out_base,'Run1/']))
                sd = ev_dir + 'Run' + str(i) + '/'
                dd = out_base + 'Run1/'
                #Rename EVs to have the correct number of runs and store in temporary direcotry
                run = []
                for l in ['_'.join([word.replace('_target',''),'Other_Content_or_Target']),'No_Interest']:
                    f_src = sd + l + '.run00' + str(i) + '.txt'
                    f_dest = dd + l + '.run001.txt'
                    shutil.copy(f_src,f_dest)
                    run.append(f_dest)
                fix_num = 1
                while (os.path.isfile(''.join([sd,word,str(fix_num),'.run00',str(i),'.txt']))):
                    f_src = sd + word + str(fix_num) + '.run00' + str(i) + '.txt'
                    f_dest = dd + word + str(fix_num) + '.run001.txt'
                    shutil.copy(f_src,f_dest)
                    run.append(f_dest)
                    tcon_target_word_fixation_gtbl = (''.join([word,str(fix_num),'>Baseline']),'T',[''.join([word,str(fix_num)])],[1])
                    target_contrasts.append(tcon_target_word_fixation_gtbl)
                    fix_num += 1
                ev_target.append(run)
                trim_vol_file = sd + 'text_end_vol_num.run00' + str(i) + '.txt'
                func_trim_point.append(int(open(trim_vol_file,'r').read()))
                f = ''.join([pardir,'fMRI_Preprocessing/Highpass/_Add_Mean_Back_In',str(i-1),'/'])
                f += os.listdir(f)[0]
                func.append(f)
                m = ''.join([pardir,'fMRI_Preprocessing/MotionPars/_realign',str(i-1),'/'])
                m += os.listdir(m)[0]
                mp.append(m)
        
        if mp and not os.path.isdir(''.join([pardir,'fMRI_Analyses/Conceptual_Change/',word.capitalize().replace('_target',''),str(runnum)])):           
            #Run analysis                      
            analysis = Workflow(name = "Target_GLM_Level1")
            data = Node(ID(fields=['func','mp','ev_target','target_contrasts',
                                   'func_trim_point','mask'],
                           mandatory_inputs=False),name='Data')
            data.inputs.func = func
            data.inputs.mp = mp
            data.inputs.ev_target = [ev_target]
            #data.inputs.ev_fixation_category = [ev_fixation_category]
            data.inputs.target_contrasts = target_contrasts
            #data.inputs.fixation_category_contrasts = fixation_category_contrasts
            data.inputs.func_trim_point = func_trim_point
            data.inputs.mask = '/gpfs/group/pul8/default/read/MNI152_T1_3mm3mm4mm_brain_mask.nii.gz'
            
            #Use the DataSink function to store all outputs
            datasink = Node(nio.DataSink(), name= 'Output')
            datasink.inputs.base_directory = pardir + 'fMRI_Analyses/Conceptual_Change/' + word.capitalize().replace('_target','') + str(runnum)
            
            question_trimmer = MapNode(fsl.ExtractROI(t_min=0),
                                  name='Question_Trimmer',
                                  iterfield=['in_file','t_size'])
                                  
            analysis.connect(data,'func',question_trimmer,'in_file')
            analysis.connect(data,'func_trim_point',question_trimmer,'t_size')
            #analysis.connect(question_trimmer,'roi_file',datasink,'Trimmed_Func')
            
            mp_trimmer = MapNode(Function(input_names=['mp','ftp'],
                                                output_names=['trimmed'],
                                                function=motionpar_trimmer),
                                                name='Motion_Parameter_Trimmer',
                                                iterfield=['mp','ftp'])
            analysis.connect(data,'mp',mp_trimmer,'mp')
            analysis.connect(data,'func_trim_point',mp_trimmer,'ftp')
            analysis.connect(mp_trimmer,'trimmed',datasink,'Trimmed_MotionPars')
            
            #Now we will model the task using FEATModel to convolve the task regressors
            #with a double gamma hemodynamic response function
            target_modelspec = MapNode(interface=model.SpecifyModel(high_pass_filter_cutoff = 128.00,
                                                          input_units = 'secs',
                                                          time_repetition=0.400), 
                                name="Target_ModelSpec",
                                iterfield=['event_files','functional_runs','realignment_parameters'])
            
            analysis.connect(data,'ev_target',target_modelspec,'event_files')
            analysis.connect(question_trimmer,('roi_file',nest_list), target_modelspec,'functional_runs')
            analysis.connect(mp_trimmer,('trimmed',nest_list),target_modelspec,'realignment_parameters')
            
            target_level1design = MapNode(interface=fsl.Level1Design(bases = {'dgamma': {'derivs':False}},
                                                              interscan_interval=.400,
                                                              model_serial_correlations=True),
                                   name="Target_level1design",
                                   iterfield=['session_info','contrasts'])
            analysis.connect(target_modelspec,'session_info',target_level1design,'session_info')
            analysis.connect(data,'target_contrasts',target_level1design,'contrasts')
            
            target_FEATModel = MapNode(interface=fsl.model.FEATModel(),
                                name = 'Target_FEATModel',
                                iterfield=['ev_files','fsf_file'])
            analysis.connect(target_level1design,('ev_files',unnest_list),target_FEATModel,'ev_files')
            analysis.connect(target_level1design,('fsf_files',unnest_list),target_FEATModel,'fsf_file')
            analysis.connect(target_FEATModel,'design_file',datasink,'Target_Task_Design')
            analysis.connect(target_FEATModel,'design_cov',datasink,'Target_Covariance_Matrix')
            analysis.connect(target_FEATModel,'design_image',datasink,'Target_Design_Matrix')
            analysis.connect(target_FEATModel,'con_file',datasink,'Target_TCons')
            #analysis.connect(target_FEATModel,'fcon_file',datasink,'Target_FCons')
            
            target_filmgls = MapNode(interface=fsl.model.FILMGLS(threshold=1000,smooth_autocorr=True),
                                         name = "Target_Fit_Model",
                                         iterfield = ['in_file','design_file','tcon_file'])#,'fcon_file'])
            analysis.connect(target_FEATModel, 'design_file',target_filmgls,'design_file')
            analysis.connect(question_trimmer,'roi_file',target_filmgls,'in_file')
            analysis.connect(target_FEATModel,'con_file',target_filmgls,'tcon_file')
            analysis.connect(target_FEATModel,'fcon_file',target_filmgls,'fcon_file')
            analysis.connect(target_filmgls,'copes',datasink,'Target_Copes')
            analysis.connect(target_filmgls,'dof_file',datasink,'Target_DOF')
            analysis.connect(target_filmgls,'fstats',datasink,'Target_Fstats')
            analysis.connect(target_filmgls,'param_estimates',datasink,'Target_Param_Estimates')
            analysis.connect(target_filmgls,'residual4d',datasink,'Target_Residual4D')
            analysis.connect(target_filmgls,'sigmasquareds',datasink,'Target_Sigma_Squareds')
            analysis.connect(target_filmgls,'thresholdac',datasink,'Target_AC_Params')
            analysis.connect(target_filmgls,'tstats',datasink,'Target_Tstats')
            analysis.connect(target_filmgls,'varcopes',datasink,'Target_Varcopes')
            #analysis.connect(target_filmgls,'zfstats',datasink,'Target_Z_Fstats')
            analysis.connect(target_filmgls,'zstats',datasink,'Target_Z_Tstats')
            
            analysis.run(plugin='MultiProc', plugin_args={'n_procs' : 5})      
