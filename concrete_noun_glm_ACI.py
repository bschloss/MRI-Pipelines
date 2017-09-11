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
import exceptions

def get_crash(crashdir):
    import os
    import gzip as gz
    import pickle as pkl
    crash = [crashdir + f for f in os.listdir(crashdir) if 'crash' in f]
    return [pkl.load(gz.open(f)) for f in crash]
    
def list_of_lists(in_file):
    return [[item] for item in in_file]

def nest_list(in_file):
    if len(in_file) == 1:
        out = in_file
    else:
        out = [in_file]
    print in_file
    print out
    return out
    
def unnest_list(in_file):
    out = []
    if len(in_file[0]) == 1:
        out = in_file
    for l in in_file:
        out = in_file[0]
    print in_file
    print out
    return out
    
def sort_copes(copes):
    output = [[] for i in range(len(copes[0]))]
    for i in range(len(copes[0])):
        for j in range(len(copes)):
            output[i].append(copes[j][i])
    return output
    
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
conc    =    ['axial',
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
        
conc = ['_'.join([word,'conc']) for word in conc]
tcon_no_interest_gtbl = ('No_Interest>Baseline','T',['No_Interest'],[1]) 
ev_dir = pardir + 'EV2/'
for word in conc:
    tcon_nonconc_content_fixation_gtbl = ('_'.join([word.replace('_conc',''),'Other_Content_or_Target>Baseline']),'T',['_'.join([word.replace('_conc',''),'Other_Content_or_Target'])],[1]) 
    tcon_conc_word_fixation_gtbl = (''.join([word,'>Baseline']),'T',[word],[1])
    concrete_contrasts = [tcon_nonconc_content_fixation_gtbl,
                          tcon_no_interest_gtbl,
                          tcon_conc_word_fixation_gtbl]    
    
    func = []
    mp = []
    ev_concrete = []
    func_trim_point = []
    concrete_copes = []
    concrete_varcopes = []
    
    #Create a temporary directory
    out_base = ''
    while out_base == '':
        letters = string.ascii_letters
        out_base = ''.join(['/tmp/tmp',''.join([letters[i] for i in random.choice(len(letters),10)])])
        out_base += '/Rename_EVs/'
        try:
            os.makedirs(out_base)
        except:
            exceptions.OSError
            out_base = ''
            
    runnum = 0       
    for i in range(1,6):
        #Locate Original EV files
        ev_file = pardir + 'EV2/Run' + str(i) + '/' + word + '.run00' + str(i) + '.txt'
        #Only include runs where the participants fixated on that word
        if not (pardir[-5:-1]=='/021' and i == 4) and not (pardir[-5:-1] == '2005' and i == 2)  and not (pardir[-5:-1] in ['2006','2014'] and i == 5):
            if not open(ev_file,'r').read() == '0\t0\t0':
	        runnum += 1
                os.makedirs(''.join([out_base,'Run',str(runnum),'/']))
                sd = ev_dir + 'Run' + str(i) + '/'
                dd = out_base + 'Run' + str(runnum) + '/'
                #Rename EVs to have the correct number of runs and store in temporary direcotry
                run = []
                for l in ['_'.join([word.replace('_conc',''),'Other_Content_or_Target']),'No_Interest',word]:
            	    f_src = sd + l + '.run00' + str(i) + '.txt'
                    f_dest = dd + l + '.run00' + str(runnum) + '.txt'
                    shutil.copy(f_src,f_dest)
            	    run.append(f_dest)
                ev_concrete.append(run)
                trim_vol_file = sd + 'text_end_vol_num.run00' + str(i) + '.txt'
                func_trim_point.append(int(open(trim_vol_file,'r').read()))
                f = ''.join([pardir,'fMRI_Preprocessing/Highpass/_Add_Mean_Back_In',str(i-1),'/'])
                f += os.listdir(f)[0]
                func.append(f)
                m = ''.join([pardir,'fMRI_Preprocessing/MotionPars/_realign',str(i-1),'/'])
                m += os.listdir(m)[0]
                mp.append(m)
            
    if mp and not os.path.isdir(''.join([pardir,'fMRI_Analyses/RSA/',word.capitalize().replace('_conc','')])):
        #Run analysis                      
        analysis = Workflow(name = "Concrete_GLM_Level1")
        data = Node(ID(fields=['func','mp','ev_concrete','concrete_contrasts',
                               'func_trim_point','mask'],
                       mandatory_inputs=False),name='Data')
        data.inputs.func = func
        data.inputs.mp = mp
        data.inputs.ev_concrete = [ev_concrete]
        #data.inputs.ev_fixation_category = [ev_fixation_category]
        data.inputs.concrete_contrasts = concrete_contrasts
        #data.inputs.fixation_category_contrasts = fixation_category_contrasts
        data.inputs.func_trim_point = func_trim_point
        data.inputs.mask = '/gpfs/group/pul8/default/read/MNI152_T1_3mm3mm4mm_brain_mask.nii.gz'
        
        #Use the DataSink function to store all outputs
        datasink = Node(nio.DataSink(), name= 'Output')
        datasink.inputs.base_directory = pardir + 'fMRI_Analyses/RSA/' + word.capitalize().replace('_conc','')
        
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
        concrete_modelspec = MapNode(interface=model.SpecifyModel(high_pass_filter_cutoff = 128.00,
                                                      input_units = 'secs',
                                                      time_repetition=0.400), 
                            name="Concrete_ModelSpec",
                            iterfield=['event_files','functional_runs','realignment_parameters'])
        
        analysis.connect(data,'ev_concrete',concrete_modelspec,'event_files')
        analysis.connect(question_trimmer,('roi_file',nest_list), concrete_modelspec,'functional_runs')
        analysis.connect(mp_trimmer,('trimmed',nest_list),concrete_modelspec,'realignment_parameters')
        
        concrete_level1design = MapNode(interface=fsl.Level1Design(bases = {'dgamma': {'derivs':False}},
                                                          interscan_interval=.400,
                                                          model_serial_correlations=True),
                               name="Concrete_level1design",
                               iterfield=['session_info','contrasts'])
        analysis.connect(concrete_modelspec,'session_info',concrete_level1design,'session_info')
        analysis.connect(data,'concrete_contrasts',concrete_level1design,'contrasts')
        
        concrete_FEATModel = MapNode(interface=fsl.model.FEATModel(),
                            name = 'Concrete_FEATModel',
                            iterfield=['ev_files','fsf_file'])
        analysis.connect(concrete_level1design,('ev_files',unnest_list),concrete_FEATModel,'ev_files')
        analysis.connect(concrete_level1design,('fsf_files',unnest_list),concrete_FEATModel,'fsf_file')
        analysis.connect(concrete_FEATModel,'design_file',datasink,'Concrete_Task_Design')
        analysis.connect(concrete_FEATModel,'design_cov',datasink,'Concrete_Covariance_Matrix')
        analysis.connect(concrete_FEATModel,'design_image',datasink,'Concrete_Design_Matrix')
        analysis.connect(concrete_FEATModel,'con_file',datasink,'Concrete_TCons')
        #analysis.connect(concrete_FEATModel,'fcon_file',datasink,'Concrete_FCons')
        
        concrete_filmgls = MapNode(interface=fsl.model.FILMGLS(threshold=1000,smooth_autocorr=True),
                                     name = "Concrete_Fit_Model",
                                     iterfield = ['in_file','design_file','tcon_file'])#,'fcon_file'])
        analysis.connect(concrete_FEATModel, 'design_file',concrete_filmgls,'design_file')
        analysis.connect(question_trimmer,'roi_file',concrete_filmgls,'in_file')
        analysis.connect(concrete_FEATModel,'con_file',concrete_filmgls,'tcon_file')
        analysis.connect(concrete_FEATModel,'fcon_file',concrete_filmgls,'fcon_file')
        analysis.connect(concrete_filmgls,'copes',datasink,'Concrete_Copes')
        analysis.connect(concrete_filmgls,'dof_file',datasink,'Concrete_DOF')
        analysis.connect(concrete_filmgls,'fstats',datasink,'Concrete_Fstats')
        analysis.connect(concrete_filmgls,'param_estimates',datasink,'Concrete_Param_Estimates')
        analysis.connect(concrete_filmgls,'residual4d',datasink,'Concrete_Residual4D')
        analysis.connect(concrete_filmgls,'sigmasquareds',datasink,'Concrete_Sigma_Squareds')
        analysis.connect(concrete_filmgls,'thresholdac',datasink,'Concrete_AC_Params')
        analysis.connect(concrete_filmgls,'tstats',datasink,'Concrete_Tstats')
        analysis.connect(concrete_filmgls,'varcopes',datasink,'Concrete_Varcopes')
        #analysis.connect(concrete_filmgls,'zfstats',datasink,'Concrete_Z_Fstats')
        analysis.connect(concrete_filmgls,'zstats',datasink,'Concrete_Z_Tstats')
        
        
        concrete_level2model = Node(fsl.model.L2Model(num_copes = len(mp)),
                                         name = 'Concrete_L2Model')
        #analysis.connect(data,'numcopes',concrete_level2model,'num_copes')
        analysis.connect(concrete_level2model,'design_con',datasink,'Concrete_L2_TCons')
        analysis.connect(concrete_level2model,'design_mat',datasink,'Concrete_L2_Design')
        analysis.connect(concrete_level2model,'design_grp',datasink,'Concrete_L2_Group')
        
        
        concrete_copemerge = MapNode(interface=fsl.Merge(dimension='t'),
                                  iterfield=['in_files'],
                                  name="Concrete_Cope_Merge")
        analysis.connect(concrete_filmgls,('copes',sort_copes),concrete_copemerge,'in_files')
        analysis.connect(concrete_copemerge,'merged_file',datasink,'Concrete_Merged_Copes')
        
        concrete_varcopemerge = MapNode(interface=fsl.Merge(dimension='t'),
                               iterfield=['in_files'],
                               name="Concrete_Varcope_Merge")
        analysis.connect(concrete_filmgls,('varcopes',sort_copes),concrete_varcopemerge,'in_files')
        analysis.connect(concrete_varcopemerge,'merged_file',datasink,'Concrete_Merged_Varcopes')
        
        concrete_flameo = MapNode(interface=fsl.model.FLAMEO(run_mode = 'fe'),
                                                               name = 'Concrete_Fixed_Effects',
                                                               iterfield = ['cope_file','var_cope_file'])#],'dof_var_cope_file'])  
        analysis.connect(concrete_copemerge,'merged_file',concrete_flameo,'cope_file')
        analysis.connect(concrete_varcopemerge,'merged_file',concrete_flameo,'var_cope_file')
        analysis.connect(concrete_level2model,'design_con',concrete_flameo,'t_con_file')
        analysis.connect(concrete_level2model,'design_mat',concrete_flameo,'design_file')
        analysis.connect(concrete_level2model,'design_grp',concrete_flameo,'cov_split_file')
        analysis.connect(data,'mask',concrete_flameo,'mask_file')
        #analysis.connect(parametric_filmgls,'dof_file',parametric_flameo,'dof_var_cope_file')
        analysis.connect(concrete_flameo,'copes',datasink,'Concrete_Fixed_Effects_Copes')
        analysis.connect(concrete_flameo,'res4d',datasink,'Concrete_Fixed_Effects_Residuals')
        analysis.connect(concrete_flameo,'tdof',datasink,'Concrete_Fixed_Effects_TDOF')
        analysis.connect(concrete_flameo,'tstats',datasink,'Concrete_Fixed_Effects_Tstats')
        analysis.connect(concrete_flameo,'var_copes',datasink,'Concrete_Fixed_Effects_Varcopes')
        analysis.connect(concrete_flameo,'zstats',datasink,'Concrete_Fixed_Effects_Z_Tstats')
        
        try:
		analysis.run(plugin='MultiProc', plugin_args={'n_procs' : 5})
	except:
		exceptions.OSError      
