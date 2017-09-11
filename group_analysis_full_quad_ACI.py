# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 11:51:36 2016

@author: bschloss
"""
import os
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface as ID
from nipype.interfaces.utility import Function

def get_crash(crashdir):
    import os
    import gzip as gz
    import pickle as pkl
    crash = [crashdir + f for f in os.listdir(crashdir) if 'crash' in f]
    return [pkl.load(gz.open(f)) for f in crash]
    
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
    
def get_dofs(imgs):
    import nibabel as nib
    import numpy as np
    data = [nib.load(img).get_data().ravel() for img in imgs]
    data = [int(np.mean([f for f in d if f > 0])) for d in data]
    return data

datadir = '/gpfs/group/pul8/default/read/'
pardirs = ['2101','2102','2003','2104','2005','2006','2007','2008','2009','2010',
           '2011','2012','2013','2014','2015','2016','2017','2218','2019','2020',
           '2021','2022','2123','2024','2025','2026','2027','2128']
           
struct = [datadir + pd + '/fMRI_Preprocessing/Registered_s2MNI_NL_Warped/_Registration_s2MNI_NL0/' for pd in pardirs]
struct = [sd + os.listdir(sd)[0] for sd in struct]
pardirs = [datadir + d + '/fMRI_Analyses/Full_Quad/' for d in pardirs]
parcopes = []
parvarcopes = []
pardof = []

for i in range(21):
    parc = [pd + 'Parametric_Fixed_Effects_Copes/_Parametric_Fixed_Effects' + str(i) + '/' for pd in pardirs]
    parc = [d + os.listdir(d)[0] for d in parc]
    parcopes.append(parc)
    parvc  = [pd + 'Parametric_Fixed_Effects_Varcopes/_Parametric_Fixed_Effects' + str(i) + '/' for pd in pardirs]
    parvc = [d + os.listdir(d)[0] for d in parvc]
    parvarcopes.append(parvc)
    pardf = [pd + 'Parametric_Fixed_Effects_TDOF/_Parametric_Fixed_Effects' + str(i) + '/' for pd in pardirs]
    pardf = [d + os.listdir(d)[0] for d in pardf]
    pardof.append(pardf)
    
analysis = Workflow(name = "fMRI_Analyses")

data = Node(ID(fields=['parvarcopes','parcopes','pardof','mask','struct'],mandatory_inputs=False),name='Data')
data.inputs.parvarcopes = parvarcopes
data.inputs.parcopes = parcopes
data.inputs.pardof = pardof
data.inputs.mask = '/gpfs/group/pul8/default/read/MNI152_T1_3mm3mm4mm_brain_mask.nii.gz'
data.struct = [struct]

#Use the DataSink function to store all outputs
datasink = Node(nio.DataSink(), name= 'Output')
datasink.inputs.base_directory = datadir + '/Group_Analyses/Full_Quad_Bilingual/'

merge_struct = Node(fsl.Merge(dimension = 't',in_files=struct), name = 'Struct_Merge')
analysis.connect(merge_struct,'merged_file',datasink,'Merged_Struct')

avg_struct = Node(interface=fsl.MeanImage(),
                        name = 'Average_Struct')
analysis.connect(merge_struct,'merged_file',avg_struct,'in_file')
analysis.connect(avg_struct,'out_file',datasink,'Average_Struct')

#Mask the image with the premade MNI mask
sub_temp = Node(fsl.ApplyMask(out_file='subject_template.nii.gz'),name='Subject_Template')
analysis.connect(data,'mask',sub_temp,'mask_file')
analysis.connect(avg_struct,'out_file',sub_temp,'in_file')
analysis.connect(sub_temp,'out_file',datasink,'Subject_Template')

group_model = Node(fsl.model.L2Model(num_copes = len(pardirs)),
                                 name = 'Parametric_L2Model')
analysis.connect(group_model,'design_con',datasink,'Group_Model_TCons')
analysis.connect(group_model,'design_mat',datasink,'Group_Model_Design')
analysis.connect(group_model,'design_grp',datasink,'Group_Model_Group')

parametric_copemerge = MapNode(interface=fsl.Merge(dimension='t'),
                          iterfield=['in_files'],
                          name="Parametric_Cope_Merge")
analysis.connect(data,'parcopes',parametric_copemerge,'in_files')
analysis.connect(parametric_copemerge,'merged_file',datasink,'Parametric_Merged_Copes')

parametric_varcopemerge = MapNode(interface=fsl.Merge(dimension='t'),
                       iterfield=['in_files'],
                       name="Parametric_Varcope_Merge")
analysis.connect(data,'parvarcopes',parametric_varcopemerge,'in_files')
analysis.connect(parametric_varcopemerge,'merged_file',datasink,'Parametric_Merged_Varcopes')

parametric_flameo = MapNode(interface=fsl.model.FLAMEO(run_mode = 'flame1'),
                                                       #sigma_dofs = 8),
                                                       name = 'Parametric_Mixed_Effects',
                                                       iterfield = ['cope_file','var_cope_file'])#],'dof_var_cope_file'])  
analysis.connect(parametric_copemerge,'merged_file',parametric_flameo,'cope_file')
analysis.connect(parametric_varcopemerge,'merged_file',parametric_flameo,'var_cope_file')
analysis.connect(group_model,'design_con',parametric_flameo,'t_con_file')
analysis.connect(group_model,'design_mat',parametric_flameo,'design_file')
analysis.connect(group_model,'design_grp',parametric_flameo,'cov_split_file')
analysis.connect(data,'mask',parametric_flameo,'mask_file')
analysis.connect(parametric_flameo,'copes',datasink,'Parametric_Mixed_Effects_Copes')
analysis.connect(parametric_flameo,'res4d',datasink,'Parametric_Mixed_Effects_Residuals')
analysis.connect(parametric_flameo,'tdof',datasink,'Parametric_Mixed_Effects_TDOF')
analysis.connect(parametric_flameo,'tstats',datasink,'Parametric_Mixed_Effects_Tstats')
analysis.connect(parametric_flameo,'var_copes',datasink,'Parametric_Mixed_Effects_Varcopes')
analysis.connect(parametric_flameo,'zstats',datasink,'Parametric_Mixed_Effects_Z_Tstats')
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

parametric_cluster_no_pthresh = MapNode(interface=fsl.model.Cluster(minclustersize=True,
                                                         out_localmax_txt_file = True,
                                                         out_localmax_vol_file = True,
                                                         out_index_file=True,
                                                         out_threshold_file = True,
                                                         out_pval_file = True),
                                        name = 'Parametric_Cluster_No_PThreshold',
                                        iterfield = ['in_file'],
                                        iterables = ('threshold',[2.33,3.08,3.27,3.62]))
analysis.connect(parametric_flameo,'zstats',parametric_cluster_no_pthresh,'in_file')
analysis.connect(parametric_cluster_no_pthresh,'index_file',datasink,'Parametric_No_PThresh_Cluster_Index')
analysis.connect(parametric_cluster_no_pthresh,'localmax_txt_file',datasink,'Parametric_No_PThresh_Cluster_LocalMax_Txt')
analysis.connect(parametric_cluster_no_pthresh,'localmax_vol_file',datasink,'Parametric_No_PThresh_Cluster_LocalMax_Vol')
analysis.connect(parametric_cluster_no_pthresh,'threshold_file',datasink,'Parametric_No_PThresh_Cluster_Threshold_File')
analysis.connect(parametric_cluster_no_pthresh,'pval_file',datasink,'Parametric_No_PThresh_Cluster_Pval_File')

analysis.run(plugin='MultiProc', plugin_args={'n_procs' : 12})
