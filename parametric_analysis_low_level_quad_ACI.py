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
    
def get_crash(crashdir):
    import os
    import gzip as gz
    import pickle as pkl
    crash = [crashdir + f for f in os.listdir(crashdir) if 'crash' in f]
    return [pkl.load(gz.open(f)) for f in crash]
    
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
ev_dir = pardir + 'EV/'

#Collect and sort event files for model
ev_parametric = []
parametric_labels = ['Content_or_Target_Word','first_instructions','No_Interest','PC1_Length_Content_or_Target',
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
    for l in parametric_labels:
        f = d + l + '.run00' + str(i) + '.txt'
        run.append(f)
    ev_parametric.append(run)

tcon_fixation_gtbl = ('Word_Fixation>Baseline','T',['Content_or_Target_Word'],[1])   
tcon_fixation_ltbl = ('Word_Fixation<Baseline','T',['Content_or_Target_Word'],[-1])   
tcon_instructions_gtbl = ('First_Instructions>Baseline','T',['first_instructions'],[1])
tcon_instructions_ltbl = ('First_Instructions<Baseline','T',['first_instructions'],[-1])
tcon_no_interest_gtbl = ('No_Interest<Baseline','T',['No_Interest'],[1])
tcon_no_interest_ltbl = ('No_Interest<Baseline','T',['No_Interest'],[-1])
tcon_length_gtbl = ('Length>Baseline','T',['Length_Content_or_Target'], [1])
tcon_length_ltbl = ('Length<Baseline','T',['Length_Content_or_Target'], [-1])
tcon_length2_gtbl = ('Length**2>Baseline','T',['Length_Content_or_Target_Squared'], [1])
tcon_length2_ltbl = ('Length**2<Baseline','T',['Length_Content_or_Target_Squared'], [-1])
tcon_length_pc_gtbl = ('Length_PC>Baseline','T',['PC1_Length_Content_or_Target'], [1])
tcon_length_pc_ltbl = ('Length_PC<Baseline','T',['PC1_Length_Content_or_Target'], [-1])
tcon_length_pc2_gtbl = ('Length_PC**2>Baseline','T',['PC1_Length_Content_or_Target_Squared'], [1])
tcon_length_pc2_ltbl = ('Length_PC**2<Baseline','T',['PC1_Length_Content_or_Target_Squared'], [-1])
tcon_concreteness_gtbl = ('Concreteness>Baseline','T',['Concreteness_Content_or_Target'], [1])
tcon_concreteness_ltbl = ('Concreteness<Baseline','T',['Concreteness_Content_or_Target'], [-1])
tcon_concreteness2_gtbl = ('Concreteness**2>Baseline','T',['Concreteness_Content_or_Target_Squared'], [1])
tcon_concreteness2_ltbl = ('Concreteness**2<Baseline','T',['Concreteness_Content_or_Target_Squared'], [-1])
tcon_frequency_gtbl = ('Frequency>Baseline','T',['Log_Freq_HAL_Content_or_Target'], [1])
tcon_frequency_ltbl = ('Frequency<Baseline','T',['Log_Freq_HAL_Content_or_Target'], [-1])
tcon_frequency2_gtbl = ('Frequency**2>Baseline','T',['Log_Freq_HAL_Content_or_Target_Squared'], [1])
tcon_frequency2_ltbl = ('Frequency**2<Baseline','T',['Log_Freq_HAL_Content_or_Target_Squared'], [-1])
tcon_OLD_gtbl = ('OLD>Baseline','T',['OLD_Content_or_Target'], [1])
tcon_OLD_ltbl = ('OLD<Baseline','T',['OLD_Content_or_Target'], [-1])
tcon_OLD2_gtbl = ('OLD**2>Baseline','T',['OLD_Content_or_Target_Squared'], [1])
tcon_OLD2_ltbl = ('OLD**2<Baseline','T',['OLD_Content_or_Target_Squared'], [-1])
tcon_PLD_gtbl = ('PLD>Baseline','T',['PLD_Content_or_Target'], [1])
tcon_PLD_ltbl = ('PLD<Baseline','T',['PLD_Content_or_Target'], [-1])
tcon_PLD2_gtbl = ('PLD**2>Baseline','T',['PLD_Content_or_Target_Squared'], [1])
tcon_PLD2_ltbl = ('PLD**2<Baseline','T',['PLD_Content_or_Target_Squared'], [-1])
tcon_position_gtbl = ('Position>Baseline','T',['Position_Content_or_Target'], [1])
tcon_position_ltbl = ('Position<Baseline','T',['Position_Content_or_Target'], [-1])
tcon_position2_gtbl = ('Position**2>Baseline','T',['Position_Content_or_Target_Squared'], [1])
tcon_position2_ltbl = ('Position**2<Baseline','T',['Position_Content_or_Target_Squared'], [-1])
tcon_AoA_gtbl = ('AoA>Baseline','T',['AoA_Content_or_Target'], [1])
tcon_AoA_ltbl = ('AoA<Baseline','T',['AoA_Content_or_Target'], [-1])
tcon_AoA2_gtbl = ('AoA**2>Baseline','T',['AoA_Content_or_Target_Squared'], [1])
tcon_AoA2_ltbl = ('AoA**2<Baseline','T',['AoA_Content_or_Target_Squared'], [-1])
tcon_nphon_gtbl = ('NPhon>Baseline','T',['NPhon_Content_or_Target'], [1])
tcon_nphon_ltbl = ('NPhon<Baseline','T',['NPhon_Content_or_Target'], [-1])
tcon_nphon2_gtbl = ('NPhon**2>Baseline','T',['NPhon_Content_or_Target_Squared'], [1])
tcon_nphon2_ltbl = ('NPhon**2<Baseline','T',['NPhon_Content_or_Target_Squared'], [-1])
tcon_nsyll_gtbl = ('NSyll>Baseline','T',['NSyll_Content_or_Target'], [1])
tcon_nsyll_ltbl = ('NSyll<Baseline','T',['NSyll_Content_or_Target'], [-1])
tcon_nsyll2_gtbl = ('NSyll**2>Baseline','T',['NSyll_Content_or_Target_Squared'], [1])
tcon_nsyll2_ltbl = ('NSyll**2<Baseline','T',['NSyll_Content_or_Target_Squared'], [-1])
                 
fcon_model_fit = ('Parametric_Model_Fit','F',[tcon_fixation_gtbl,
                                              tcon_instructions_gtbl,
                                              #tcon_no_interest_gtbl,
                                              tcon_length_pc_gtbl,
                                              tcon_length_pc2_gtbl,
                                              tcon_frequency_gtbl,
                                              tcon_frequency2_gtbl])
                                              #tcon_OLD_gtbl,
                                              #tcon_PLD_gtbl,
                                              #tcon_position_gtbl,
                                              #tcon_AoA_gtbl,
                                              #tcon_nphon_gtbl,
                                              #tcon_nsyll_gtbl,
                                              #tcon_length_gtbl,
                                              #tcon_length2_gtbl,
                                              #tcon_concreteness_gtbl,
                                              #tcon_concreteness2_gtbl,
                                              #tcon_OLD2_gtbl,
                                              #tcon_PLD2_gtbl,
                                              #tcon_position2_gtbl,
                                              #tcon_AoA2_gtbl,
                                              #tcon_nphon2_gtbl,
                                              #tcon_nsyll2_gtbl])
                                    


parametric_contrasts = [fcon_model_fit,
                        tcon_fixation_gtbl,
                        tcon_fixation_ltbl,
                        tcon_no_interest_gtbl,
                        tcon_no_interest_ltbl,
                        tcon_length_pc_gtbl,
                        tcon_length_pc_ltbl,
                        tcon_length_pc2_gtbl,
                        tcon_length_pc2_ltbl,
                        tcon_frequency_gtbl,
                        tcon_frequency_ltbl,
                        tcon_frequency2_gtbl,
                        tcon_frequency2_ltbl]
                        #tcon_length_gtbl,
                        #tcon_length_ltbl,
                        #tcon_length2_gtbl,
                        #tcon_length2_ltbl,
                        #tcon_concreteness_gtbl,
                        #tcon_concreteness_ltbl,
                        #tcon_concreteness2_gtbl,
                        #tcon_concreteness2_ltbl,
                        #tcon_OLD_gtbl,
                        #tcon_OLD_ltbl,
                        #tcon_OLD2_gtbl,
                        #tcon_OLD2_ltbl,
                        #tcon_PLD_gtbl,
                        #tcon_PLD_ltbl,
                        #tcon_PLD2_gtbl,
                        #tcon_PLD2_ltbl,
                        #tcon_position_gtbl,
                        #tcon_position_ltbl,
                        #tcon_position2_gtbl,
                        #tcon_position2_ltbl,
                        #tcon_AoA_gtbl,
                        #tcon_AoA_ltbl,
                        #tcon_AoA2_gtbl,
                        #tcon_AoA2_ltbl,
                        #tcon_nphon_gtbl,
                        #tcon_nphon_ltbl,
                        #tcon_nphon2_gtbl,
                        #tcon_nphon2_ltbl,
                        #tcon_nsyll_gtbl,
                        #tcon_nsyll_ltbl,
                        #tcon_nsyll2_gtbl,
                        #tcon_nsyll2_ltbl]
                                                             
if pardir.rstrip('/')[-4:] == '/014' or pardir.rstrip('/')[-4:] == '2006' or pardir.rstrip('/')[-4:] == '2014':
    if len(func) == 5:
        func.pop(4)
    if len(mp) == 5:
        mp.pop(4)
    if len(ev_parametric) == 5:
        ev_parametric.pop(4)

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
    func = [func + '_Question_Trimmer_Smoothed' + str(i) + '/' for i in range(4)]
    func = [d + os.listdir(d)[0] for d in func]
    mp = pardir + 'fMRI_Preprocessing/Trimmed_MotionPars/'
    mp = [mp + '_Motion_Parameter_Trimmer' + str(i) + '/' for i in range(4)] 
    mp = [d + os.listdir(d)[0] for d in mp]   
    ev_parametric.pop(1)
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
    
    
analysis = Workflow(name = "fMRI_Analyses_Low_Level_Quad")
data = Node(ID(fields=['func','mp','ev_parametric',
                       'parametric_contrasts','mask'],
               mandatory_inputs=False),name='Data')
data.inputs.func = func
data.inputs.mp = mp
data.inputs.ev_parametric = [ev_parametric]
data.inputs.parametric_contrasts = parametric_contrasts
data.inputs.mask = '/gpfs/group/pul8/default/read/MNI152_T1_3mm3mm4mm_brain_mask.nii.gz'

#Use the DataSink function to store all outputs
datasink = Node(nio.DataSink(), name= 'Output')
datasink.inputs.base_directory = pardir + 'fMRI_Analyses/Low_Level_Quad/'

#Now we will model the task using FEATModel to convolve the task regressors
#with a double gamma hemodynamic response function and the time derivative
parametric_modelspec = MapNode(interface=model.SpecifyModel(high_pass_filter_cutoff = 128.00,
                                              input_units = 'secs',
                                              time_repetition=0.400), 
                    name="Parametric_ModelSpec",
                    iterfield=['event_files','functional_runs','realignment_parameters'])

analysis.connect(data,'ev_parametric',parametric_modelspec,'event_files')
analysis.connect(data,('func',nest_list), parametric_modelspec,'functional_runs')
analysis.connect(data,('mp',nest_list),parametric_modelspec,'realignment_parameters')

parametric_level1design = MapNode(interface=fsl.Level1Design(bases = {'dgamma': {'derivs':False}},
                                                  interscan_interval=.400,
                                                  model_serial_correlations=True),
                       name="Parametric_level1design",
                       iterfield=['session_info','contrasts'])
analysis.connect(parametric_modelspec,'session_info',parametric_level1design,'session_info')
analysis.connect(data,'parametric_contrasts',parametric_level1design,'contrasts')

parametric_FEATModel = MapNode(interface=fsl.model.FEATModel(),
                    name = 'Parametric_FEATModel',
                    iterfield=['ev_files','fsf_file'])
analysis.connect(parametric_level1design,('ev_files',unnest_list),parametric_FEATModel,'ev_files')
analysis.connect(parametric_level1design,('fsf_files',unnest_list),parametric_FEATModel,'fsf_file')
analysis.connect(parametric_FEATModel,'design_file',datasink,'Parametric_Task_Design')
analysis.connect(parametric_FEATModel,'design_cov',datasink,'Parametric_Covariance_Matrix')
analysis.connect(parametric_FEATModel,'design_image',datasink,'Parametric_Design_Matrix')
analysis.connect(parametric_FEATModel,'con_file',datasink,'Parametric_TCons')
analysis.connect(parametric_FEATModel,'fcon_file',datasink,'Parametric_FCons')

parametric_filmgls = MapNode(interface=fsl.model.FILMGLS(threshold=0.00,smooth_autocorr=True),
                             name = "Parametric_Fit_Model",
                             iterfield = ['in_file','design_file','tcon_file','fcon_file'])
analysis.connect(parametric_FEATModel, 'design_file',parametric_filmgls, 'design_file')
analysis.connect(data,'func',parametric_filmgls,'in_file')
analysis.connect(parametric_FEATModel, 'con_file',parametric_filmgls,'tcon_file')
analysis.connect(parametric_FEATModel,'fcon_file',parametric_filmgls,'fcon_file')
analysis.connect(parametric_filmgls,'copes',datasink,'Parametric_Copes')
analysis.connect(parametric_filmgls,'dof_file',datasink,'Parametric_DOF')
analysis.connect(parametric_filmgls,'fstats',datasink,'Parametric_Fstats')
analysis.connect(parametric_filmgls,'param_estimates',datasink,'Parametric_Param_Estimates')
analysis.connect(parametric_filmgls,'residual4d',datasink,'Parametric_Residual4D')
analysis.connect(parametric_filmgls,'sigmasquareds',datasink,'Parametric_Sigma_Squareds')
analysis.connect(parametric_filmgls,'thresholdac',datasink,'Parametric_AC_Params')
analysis.connect(parametric_filmgls,'tstats',datasink,'Parametric_Tstats')
analysis.connect(parametric_filmgls,'varcopes',datasink,'Parametric_Varcopes')
analysis.connect(parametric_filmgls,'zfstats',datasink,'Parametric_Z_Fstats')
analysis.connect(parametric_filmgls,'zstats',datasink,'Parametric_Z_Tstats')

parametric_level2model = Node(fsl.model.L2Model(num_copes = len(mp)),
                                 name = 'Parametric_L2Model')
#analysis.connect(data,'numcopes',parametric_level2model,'num_copes')
analysis.connect(parametric_level2model,'design_con',datasink,'Parametric_L2_TCons')
analysis.connect(parametric_level2model,'design_mat',datasink,'Parametric_L2_Design')
analysis.connect(parametric_level2model,'design_grp',datasink,'Parametric_L2_Group')


parametric_copemerge = MapNode(interface=fsl.Merge(dimension='t'),
                          iterfield=['in_files'],
                          name="Parametric_Cope_Merge")
analysis.connect(parametric_filmgls,('copes',sort_copes),parametric_copemerge,'in_files')
analysis.connect(parametric_copemerge,'merged_file',datasink,'Parametric_Merged_Copes')

parametric_varcopemerge = MapNode(interface=fsl.Merge(dimension='t'),
                       iterfield=['in_files'],
                       name="Parametric_Varcope_Merge")
analysis.connect(parametric_filmgls,('varcopes',sort_copes),parametric_varcopemerge,'in_files')
analysis.connect(parametric_varcopemerge,'merged_file',datasink,'Parametric_Merged_Varcopes')
                
parametric_flameo = MapNode(interface=fsl.model.FLAMEO(run_mode = 'fe'),
                                                       name = 'Parametric_Fixed_Effects',
                                                       iterfield = ['cope_file','var_cope_file'])#],'dof_var_cope_file'])  
analysis.connect(parametric_copemerge,'merged_file',parametric_flameo,'cope_file')
analysis.connect(parametric_varcopemerge,'merged_file',parametric_flameo,'var_cope_file')
analysis.connect(parametric_level2model,'design_con',parametric_flameo,'t_con_file')
analysis.connect(parametric_level2model,'design_mat',parametric_flameo,'design_file')
analysis.connect(parametric_level2model,'design_grp',parametric_flameo,'cov_split_file')
analysis.connect(data,'mask',parametric_flameo,'mask_file')
#analysis.connect(parametric_filmgls,'dof_file',parametric_flameo,'dof_var_cope_file')
analysis.connect(parametric_flameo,'copes',datasink,'Parametric_Fixed_Effects_Copes')
analysis.connect(parametric_flameo,'res4d',datasink,'Parametric_Fixed_Effects_Residuals')
analysis.connect(parametric_flameo,'tdof',datasink,'Parametric_Fixed_Effects_TDOF')
analysis.connect(parametric_flameo,'tstats',datasink,'Parametric_Fixed_Effects_Tstats')
analysis.connect(parametric_flameo,'var_copes',datasink,'Parametric_Fixed_Effects_Varcopes')
analysis.connect(parametric_flameo,'zstats',datasink,'Parametric_Fixed_Effects_Z_Tstats')

parametric_fwe = MapNode(interface=Function(input_names=['zstat','mask'],
                                            output_names=['fwe_corrected'],
                                            function=family_wise_error),
                                    name='Parametric_FWE',
                                    iterfield=['zstat'])
analysis.connect(parametric_flameo,'zstats',parametric_fwe,'zstat')
analysis.connect(data,'mask',parametric_fwe,'mask')
analysis.connect(parametric_fwe,'fwe_corrected',datasink,'Parametric_FWE')

parametric_fdr = MapNode(interface=Function(input_names=['zstat','mask'],
                                            output_names = ['fdr_corrected'],
                                            function=false_discovery_rate),
                                   name='Parametric_FDR',
                                   iterfield=['zstat'])
analysis.connect(parametric_flameo,'zstats',parametric_fdr,'zstat')
analysis.connect(data,'mask',parametric_fdr,'mask')
analysis.connect(parametric_fdr,'fdr_corrected',datasink,'Parametric_FDR')

parametric_smooth_est = MapNode(interface=fsl.SmoothEstimate(),
                                name = "Parametric_Smooth_Estimate",
                                iterfield = ['zstat_file'])
analysis.connect(parametric_flameo,'zstats',parametric_smooth_est,'zstat_file')
analysis.connect(data,'mask',parametric_smooth_est,'mask_file')

parametric_cluster = MapNode(interface=fsl.model.Cluster(minclustersize=True,
                                                         out_localmax_txt_file = True,
                                                         out_localmax_vol_file = True,
                                                         out_index_file=True,
                                                         out_threshold_file = True,
                                                         out_pval_file = True,
                                                         pthreshold = .05),
                                        name = 'Parametric_Cluster',
                                        iterfield = ['in_file','dlh','volume'],
                                        iterables = ('threshold',[2.33,3.08,3.27,3.62]))
analysis.connect(parametric_flameo,'zstats',parametric_cluster,'in_file')
analysis.connect(parametric_smooth_est,'dlh',parametric_cluster,'dlh')
analysis.connect(parametric_smooth_est,'volume',parametric_cluster,'volume')
analysis.connect(parametric_cluster,'index_file',datasink,'Parametric_Cluster_Index')
analysis.connect(parametric_cluster,'localmax_txt_file',datasink,'Parametric_Cluster_LocalMax_Txt')
analysis.connect(parametric_cluster,'localmax_vol_file',datasink,'Parametric_Cluster_LocalMax_Vol')
analysis.connect(parametric_cluster,'threshold_file',datasink,'Parametric_Cluster_Threshold_File')
analysis.connect(parametric_cluster,'pval_file',datasink,'Parametric_Cluster_Pval_File')

analysis.write_graph(dotfilename='fMRI_Preprocessing_Graph.dot',format='svg')
analysis.write_graph(dotfilename='fMRI_Preprocessing_Graph.dot',format='svg',graph2use='exec')
analysis.run(plugin='MultiProc', plugin_args={'n_procs' : 5})