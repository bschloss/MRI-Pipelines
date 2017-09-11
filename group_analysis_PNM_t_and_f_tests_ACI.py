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

def randpar_design(numcopes,numpars):
    import os
    import numpy as np
    import subprocess as sp
    out_base = os.getcwd()
    mat = np.zeros((numcopes*numpars,numcopes+numpars),dtype=np.int)
    for i in range(numcopes):
        for j in range(numpars):
            mat[i*numpars + j][i] = 1
            mat[i*numpars + j][numcopes + j] = 1
    np.savetxt('/'.join([out_base,'design.txt']),mat)
    grp = np.ones((numcopes*numpars,1),dtype=np.int)
    con = np.zeros((numcopes,numcopes+numpars),dtype=np.int)
    fts = np.ones((1,numcopes),dtype=np.int)
    for i in range(numcopes):
        con[i,i] = 1
    np.savetxt('/'.join([out_base,'group.txt']),grp)
    np.savetxt('/'.join([out_base,'tcontrast.txt']),con)
    np.savetxt('/'.join([out_base,'fcontrast.txt']),fts)
    sp.call(['Text2Vest','/'.join([out_base,'design.txt']),'/'.join([out_base,'Random_Effects_PNM.mat'])])
    sp.call(['Text2Vest','/'.join([out_base,'group.txt']),'/'.join([out_base,'Random_Effects_PNM.grp'])])
    sp.call(['Text2Vest','/'.join([out_base,'tcontrast.txt']),'/'.join([out_base,'Random_Effects_PNM.con'])])
    sp.call(['Text2Vest','/'.join([out_base,'fcontrast.txt']),'/'.join([out_base,'Random_Effects_PNM.fts'])])
    design_mat = '/'.join([out_base,'Random_Effects_PNM.mat'])
    design_con = '/'.join([out_base,'Random_Effects_PNM.con'])
    design_grp = '/'.join([out_base,'Random_Effects_PNM.grp'])
    design_fts = '/'.join([out_base,'Random_Effects_PNM.fts'])
    return design_mat,design_con,design_grp,design_fts

    
#--dm=design.mat --tc=design.con --cs=design.grp --runmode=flame1
datadir = '/gpfs/group/pul8/default/read/'
pardirs = ['201','002','003','004','105','006','107','008','009',
	      '011','012','013','214','015','016','017','018','019','020',
           '021','122','023','024','025','026','027','028','029','030',
	      '031','132','033','034','035','036','037','038','039','040',
	      '041','042','043','044','045',      '047','048','049','050']
struct = [datadir + pd + '/Rest_Preprocessing/Registered_s2MNI_NL_Warped/_Registration_s2MNI_NL0/' for pd in pardirs]
struct = [sd + os.listdir(sd)[0] for sd in struct]
pardirs = [datadir + d + '/Rest_Preprocessing/' for d in pardirs]
pnmcopes = []
pnmvarcopes = []

for i in range(36):
    pnmc = [pd + 'Copes_F2MNI_NL_Warped/_Copes_Warper_F2MNI' + str(i) + '/' for pd in pardirs]
    pnmc = [d + os.listdir(d)[0] for d in pnmc]
    pnmcopes.append(pnmc)
    pnmvc  = [pd + 'Varcopes_F2MNI_NL_Warped/_Varcopes_Warper_F2MNI' + str(i) + '/' for pd in pardirs]
    pnmvc = [d + os.listdir(d)[0] for d in pnmvc]
    pnmvarcopes.append(pnmvc)
    
analysis = Workflow(name = "PNM_Group_Analyses")

data = Node(ID(fields=['pnmvarcopes','pnmcopes','mask','struct','numpars','numcopes'],mandatory_inputs=False),name='Data')
data.inputs.pnmvarcopes = pnmvarcopes
data.inputs.pnmcopes = pnmcopes
data.inputs.mask = '/gpfs/group/pul8/default/read/MNI152_T1_3mm3mm4mm_brain_mask.nii.gz'
data.struct = [struct]
data.numpars = len(pardirs)
data.numcopes = 36

#Use the DataSink function to store all outputs
datasink = Node(nio.DataSink(), name= 'Output')
datasink.inputs.base_directory = datadir + '/Group_Analyses/PNM/'

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

group_model = Node(interface=Function(input_names=['numcopes','numpars'],
                                      output_names=['design_mat','design_con','design_grp','design_fts'],
                                      function=randpar_design),
                   name = 'PNM_Random_Effects_Design')
analysis.connect(data,'numcopes',group_model,'numcopes')
analysis.connect(data,'numpars',group_model,'numpars')
analysis.connect(group_model,'design_con',datasink,'Group_Model_TCons')
analysis.connect(group_model,'design_mat',datasink,'Group_Model_Design')
analysis.connect(group_model,'design_grp',datasink,'Group_Model_Group')
analysis.connect(group_model,'design_fts',datasink,'Group_Model_FCons')

pnm_copemerge = MapNode(interface=fsl.Merge(dimension='t'),
                          iterfield=['in_files'],
                          name="PNM_Cope_Merge")
analysis.connect(data,'pnmcopes',pnm_copemerge,'in_files')
analysis.connect(pnm_copemerge,'merged_file',datasink,'PNM_Merged_Copes')

pnm_copemergemerge = Node(interface=fsl.Merge(dimension='t'),
                          name='PNM_Cope_Merge_Merge')
analysis.connect(pnm_copemerge,'merged_file',pnm_copemergemerge,'in_files')
analysis.connect(pnm_copemergemerge,'merged_file',datasink,'PNM_Merged_Merged_Copes')

pnm_varcopemerge = MapNode(interface=fsl.Merge(dimension='t'),
                       iterfield=['in_files'],
                       name="PNM_Varcope_Merge")
analysis.connect(data,'pnmvarcopes',pnm_varcopemerge,'in_files')
analysis.connect(pnm_varcopemerge,'merged_file',datasink,'PNM_Merged_Varcopes')

pnm_varcopemergemerge = Node(interface=fsl.Merge(dimension='t'),
                          name='PNM_Varcope_Merge_Merge')
analysis.connect(pnm_varcopemerge,'merged_file',pnm_varcopemergemerge,'in_files')
analysis.connect(pnm_varcopemergemerge,'merged_file',datasink,'PNM_Merged_Merged_Varcopes')

pnm_flameo = MapNode(interface=fsl.model.FLAMEO(run_mode = 'flame1'),
                                                       #sigma_dofs = 8),
                                                       name = 'PNM_Mixed_Effects',
                                                       iterfield = ['cope_file','var_cope_file'])#],'dof_var_cope_file'])  
analysis.connect(pnm_copemergemerge,'merged_file',pnm_flameo,'cope_file')
analysis.connect(pnm_varcopemergemerge,'merged_file',pnm_flameo,'var_cope_file')
analysis.connect(group_model,'design_con',pnm_flameo,'t_con_file')
analysis.connect(group_model,'design_mat',pnm_flameo,'design_file')
analysis.connect(group_model,'design_grp',pnm_flameo,'cov_split_file')
analysis.connect(group_model,'design_fts',pnm_flameo,'f_con_file')
analysis.connect(data,'mask',pnm_flameo,'mask_file')
analysis.connect(pnm_flameo,'copes',datasink,'PNM_Mixed_Effects_Copes')
analysis.connect(pnm_flameo,'res4d',datasink,'PNM_Mixed_Effects_Residuals')
analysis.connect(pnm_flameo,'tdof',datasink,'PNM_Mixed_Effects_TDOF')
analysis.connect(pnm_flameo,'tstats',datasink,'PNM_Mixed_Effects_Tstats')
analysis.connect(pnm_flameo,'var_copes',datasink,'PNM_Mixed_Effects_Varcopes')
analysis.connect(pnm_flameo,'zstats',datasink,'PNM_Mixed_Effects_Z_Tstats')

pnm_fwe = MapNode(interface=Function(input_names=['zstat','mask'],
                                            output_names=['fwe_corrected'],
                                            function=family_wise_error),
                                    name='PNM_FWE',
                                    iterfield=['zstat'])
analysis.connect(pnm_flameo,'zstats',pnm_fwe,'zstat')
analysis.connect(data,'mask',pnm_fwe,'mask')
analysis.connect(pnm_fwe,'fwe_corrected',datasink,'PNM_FWE')

pnm_fdr = MapNode(interface=Function(input_names=['zstat','mask'],
                                            output_names = ['fdr_corrected'],
                                            function=false_discovery_rate),
                                   name='PNM_FDR',
                                   iterfield=['zstat'])
analysis.connect(pnm_flameo,'zstats',pnm_fdr,'zstat')
analysis.connect(data,'mask',pnm_fdr,'mask')
analysis.connect(pnm_fdr,'fdr_corrected',datasink,'PNM_FDR')

pnm_smooth_est = MapNode(interface=fsl.SmoothEstimate(),
                                name = "PNM_Smooth_Estimate",
                                iterfield = ['zstat_file'])
analysis.connect(pnm_flameo,'zstats',pnm_smooth_est,'zstat_file')
analysis.connect(data,'mask',pnm_smooth_est,'mask_file')

pnm_cluster = MapNode(interface=fsl.model.Cluster(minclustersize=True,
                                                         out_localmax_txt_file = True,
                                                         out_localmax_vol_file = True,
                                                         out_index_file=True,
                                                         out_threshold_file = True,
                                                         out_pval_file = True,
                                                         pthreshold = .05),
                                        name = 'PNM_Cluster',
                                        iterfield = ['in_file','dlh','volume'],
                                        iterables = ('threshold',[2.33,3.08,3.27,3.62]))
analysis.connect(pnm_flameo,'zstats',pnm_cluster,'in_file')
analysis.connect(pnm_smooth_est,'dlh',pnm_cluster,'dlh')
analysis.connect(pnm_smooth_est,'volume',pnm_cluster,'volume')
analysis.connect(pnm_cluster,'index_file',datasink,'PNM_Cluster_Index')
analysis.connect(pnm_cluster,'localmax_txt_file',datasink,'PNM_Cluster_LocalMax_Txt')
analysis.connect(pnm_cluster,'localmax_vol_file',datasink,'PNM_Cluster_LocalMax_Vol')
analysis.connect(pnm_cluster,'threshold_file',datasink,'PNM_Cluster_Threshold_File')
analysis.connect(pnm_cluster,'pval_file',datasink,'PNM_Cluster_Pval_File')

pnm_cluster_no_pthresh = MapNode(interface=fsl.model.Cluster(minclustersize=True,
                                                         out_localmax_txt_file = True,
                                                         out_localmax_vol_file = True,
                                                         out_index_file=True,
                                                         out_threshold_file = True,
                                                         out_pval_file = True),
                                        name = 'PNM_Cluster_No_PThreshold',
                                        iterfield = ['in_file'],
                                        iterables = ('threshold',[2.33,3.08,3.27,3.62]))
analysis.connect(pnm_flameo,'zstats',pnm_cluster_no_pthresh,'in_file')
analysis.connect(pnm_cluster_no_pthresh,'index_file',datasink,'PNM_No_PThresh_Cluster_Index')
analysis.connect(pnm_cluster_no_pthresh,'localmax_txt_file',datasink,'PNM_No_PThresh_Cluster_LocalMax_Txt')
analysis.connect(pnm_cluster_no_pthresh,'localmax_vol_file',datasink,'PNM_No_PThresh_Cluster_LocalMax_Vol')
analysis.connect(pnm_cluster_no_pthresh,'threshold_file',datasink,'PNM_No_PThresh_Cluster_Threshold_File')
analysis.connect(pnm_cluster_no_pthresh,'pval_file',datasink,'PNM_No_PThresh_Cluster_Pval_File')

pnm_fwe_f = MapNode(interface=Function(input_names=['zstat','mask'],
                                            output_names=['fwe_corrected'],
                                            function=family_wise_error),
                                    name='PNM_FWE_FStats',
                                    iterfield=['zstat'])
analysis.connect(pnm_flameo,'zfstats',pnm_fwe_f,'zstat')
analysis.connect(data,'mask',pnm_fwe_f,'mask')
analysis.connect(pnm_fwe_f,'fwe_corrected',datasink,'PNM_FWE_F')

pnm_fdr_f = MapNode(interface=Function(input_names=['zstat','mask'],
                                            output_names = ['fdr_corrected'],
                                            function=false_discovery_rate),
                                   name='PNM_FDR_FStats',
                                   iterfield=['zstat'])
analysis.connect(pnm_flameo,'zfstats',pnm_fdr_f,'zstat')
analysis.connect(data,'mask',pnm_fdr_f,'mask')
analysis.connect(pnm_fdr_f,'fdr_corrected',datasink,'PNM_FDR_FStats')

pnm_smooth_est_f = MapNode(interface=fsl.SmoothEstimate(),
                                name = "PNM_Smooth_Estimate_FStats",
                                iterfield = ['zstat_file'])
analysis.connect(pnm_flameo,'zfstats',pnm_smooth_est_f,'zstat_file')
analysis.connect(data,'mask',pnm_smooth_est_f,'mask_file')

pnm_cluster_f = MapNode(interface=fsl.model.Cluster(minclustersize=True,
                                                         out_localmax_txt_file = True,
                                                         out_localmax_vol_file = True,
                                                         out_index_file=True,
                                                         out_threshold_file = True,
                                                         out_pval_file = True,
                                                         pthreshold = .05),
                                        name = 'PNM_Cluster_FStats',
                                        iterfield = ['in_file','dlh','volume'],
                                        iterables = ('threshold',[2.33,3.08,3.27,3.62]))
analysis.connect(pnm_flameo,'zfstats',pnm_cluster_f,'in_file')
analysis.connect(pnm_smooth_est_f,'dlh',pnm_cluster_f,'dlh')
analysis.connect(pnm_smooth_est_f,'volume',pnm_cluster_f,'volume')
analysis.connect(pnm_cluster_f,'index_file',datasink,'PNM_Cluster_Index_FStats')
analysis.connect(pnm_cluster_f,'localmax_txt_file',datasink,'PNM_Cluster_LocalMax_Txt_FStats')
analysis.connect(pnm_cluster_f,'localmax_vol_file',datasink,'PNM_Cluster_LocalMax_Vol_FStats')
analysis.connect(pnm_cluster_f,'threshold_file',datasink,'PNM_Cluster_Threshold_File_FStats')
analysis.connect(pnm_cluster_f,'pval_file',datasink,'PNM_Cluster_Pval_File_FStats')

pnm_cluster_no_pthresh_f = MapNode(interface=fsl.model.Cluster(minclustersize=True,
                                                         out_localmax_txt_file = True,
                                                         out_localmax_vol_file = True,
                                                         out_index_file=True,
                                                         out_threshold_file = True,
                                                         out_pval_file = True),
                                        name = 'PNM_Cluster_No_PThreshold_FStats',
                                        iterfield = ['in_file'],
                                        iterables = ('threshold',[2.33,3.08,3.27,3.62]))
analysis.connect(pnm_flameo,'zfstats',pnm_cluster_no_pthresh_f,'in_file')
analysis.connect(pnm_cluster_no_pthresh_f,'index_file',datasink,'PNM_No_PThresh_Cluster_Index_FStats')
analysis.connect(pnm_cluster_no_pthresh_f,'localmax_txt_file',datasink,'PNM_No_PThresh_Cluster_LocalMax_Txt_FStats')
analysis.connect(pnm_cluster_no_pthresh_f,'localmax_vol_file',datasink,'PNM_No_PThresh_Cluster_LocalMax_Vol_FStats')
analysis.connect(pnm_cluster_no_pthresh_f,'threshold_file',datasink,'PNM_No_PThresh_Cluster_Threshold_File_FStats')
analysis.connect(pnm_cluster_no_pthresh_f,'pval_file',datasink,'PNM_No_PThresh_Cluster_Pval_File_FStats')

analysis.run(plugin='MultiProc', plugin_args={'n_procs' : 12})
