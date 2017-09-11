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
#pardirs = ['201','002','003','004','105','006','107','008','009','110',
#           '011','012','013','214','015','016','017','018','019','020',
#           '021','122','023','024','025','026','027','028','029','030',
#           '031','132','033','034','035','036','037','038','039','040',
#           '041','042','043','044','045','046','047','048','049','050']           
struct = [datadir + pd + '/fMRI_Preprocessing/Registered_s2MNI_NL_Warped/_Registration_s2MNI_NL0/' for pd in pardirs]
struct = [sd + os.listdir(sd)[0] for sd in struct]
pardirs = [datadir + d + '/fMRI_Analyses/Low_Level_Linear_ID/' for d in pardirs]
parcopes = []
parvarcopes = []
pardof = []

for i in range(23):
    parc = [pd + 'Fixed_Effects_Copes/_Fixed_Effects' + str(i) + '/' for pd in pardirs]
    parc = [d + os.listdir(d)[0] for d in parc]
    parcopes.append(parc)
    parvc  = [pd + 'Fixed_Effects_Varcopes/_Fixed_Effects' + str(i) + '/' for pd in pardirs]
    parvc = [d + os.listdir(d)[0] for d in parvc]
    parvarcopes.append(parvc)
    pardf = [pd + 'Fixed_Effects_TDOF/_Fixed_Effects' + str(i) + '/' for pd in pardirs]
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
datasink.inputs.base_directory = datadir + '/Group_Analyses/Low_Level_Linear_Bilingual_ID/'

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
                                 name = 'L2Model')
analysis.connect(group_model,'design_con',datasink,'Group_Model_TCons')
analysis.connect(group_model,'design_mat',datasink,'Group_Model_Design')
analysis.connect(group_model,'design_grp',datasink,'Group_Model_Group')

copemerge = MapNode(interface=fsl.Merge(dimension='t'),
                          iterfield=['in_files'],
                          name="Cope_Merge")
analysis.connect(data,'parcopes',copemerge,'in_files')
analysis.connect(copemerge,'merged_file',datasink,'Merged_Copes')

varcopemerge = MapNode(interface=fsl.Merge(dimension='t'),
                       iterfield=['in_files'],
                       name="Varcope_Merge")
analysis.connect(data,'parvarcopes',varcopemerge,'in_files')
analysis.connect(varcopemerge,'merged_file',datasink,'Merged_Varcopes')

flameo = MapNode(interface=fsl.model.FLAMEO(run_mode = 'flame1'),
                                                       #sigma_dofs = 8),
                                                       name = 'Mixed_Effects',
                                                       iterfield = ['cope_file','var_cope_file'])#],'dof_var_cope_file'])  
analysis.connect(copemerge,'merged_file',flameo,'cope_file')
analysis.connect(varcopemerge,'merged_file',flameo,'var_cope_file')
analysis.connect(group_model,'design_con',flameo,'t_con_file')
analysis.connect(group_model,'design_mat',flameo,'design_file')
analysis.connect(group_model,'design_grp',flameo,'cov_split_file')
analysis.connect(data,'mask',flameo,'mask_file')
analysis.connect(flameo,'copes',datasink,'Mixed_Effects_Copes')
analysis.connect(flameo,'res4d',datasink,'Mixed_Effects_Residuals')
analysis.connect(flameo,'tdof',datasink,'Mixed_Effects_TDOF')
analysis.connect(flameo,'tstats',datasink,'Mixed_Effects_Tstats')
analysis.connect(flameo,'var_copes',datasink,'Mixed_Effects_Varcopes')
analysis.connect(flameo,'zstats',datasink,'Mixed_Effects_Z_Tstats')
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

cluster_no_pthresh = MapNode(interface=fsl.model.Cluster(minclustersize=True,
                                                         out_localmax_txt_file = True,
                                                         out_localmax_vol_file = True,
                                                         out_index_file=True,
                                                         out_threshold_file = True,
                                                         out_pval_file = True),
                                        name = 'Cluster_No_PThreshold',
                                        iterfield = ['in_file'],
                                        iterables = ('threshold',[2.33,3.08,3.27,3.62]))
analysis.connect(flameo,'zstats',cluster_no_pthresh,'in_file')
analysis.connect(cluster_no_pthresh,'index_file',datasink,'No_PThresh_Cluster_Index')
analysis.connect(cluster_no_pthresh,'localmax_txt_file',datasink,'No_PThresh_Cluster_LocalMax_Txt')
analysis.connect(cluster_no_pthresh,'localmax_vol_file',datasink,'No_PThresh_Cluster_LocalMax_Vol')
analysis.connect(cluster_no_pthresh,'threshold_file',datasink,'No_PThresh_Cluster_Threshold_File')
analysis.connect(cluster_no_pthresh,'pval_file',datasink,'No_PThresh_Cluster_Pval_File')


analysis.run(plugin='MultiProc', plugin_args={'n_procs' : 10})
