# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 12:17:12 2017

@author: bschloss
"""
import os
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
from nipype.interfaces.utility import IdentityInterface as ID
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import Function
import argparse as ap
import exceptions

def unnest_list(l):
	return l[0]

def PNM_Feat(parfsf,rest): #it takes the participant-specific file of how the model should be run and the rest data
    import os
    import subprocess as sp
    import shutil as sh
    out_base = os.getcwd()
    os.chdir(parfsf.rstrip('design.fsf'))
    p = sp.Popen(['feat_model','design'])
    p.wait()
    p = sp.Popen(['film_gls',
                  ''.join(['--in=',rest]),
                  '--rn=stats',
                  '--pd=design.mat',
                  '--thr=0.000000',
                  '--sa',
                  '--con=design.con',
                  '--ven=3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36',
                  '--vef=designVoxelwiseEV3,designVoxelwiseEV4,designVoxelwiseEV5,designVoxelwiseEV6,designVoxelwiseEV7,designVoxelwiseEV8,designVoxelwiseEV9,designVoxelwiseEV10,designVoxelwiseEV11,designVoxelwiseEV12,designVoxelwiseEV13,designVoxelwiseEV14,designVoxelwiseEV15,designVoxelwiseEV16,designVoxelwiseEV17,designVoxelwiseEV18,designVoxelwiseEV19,designVoxelwiseEV20,designVoxelwiseEV21,designVoxelwiseEV22,designVoxelwiseEV23,designVoxelwiseEV24,designVoxelwiseEV25,designVoxelwiseEV26,designVoxelwiseEV27,designVoxelwiseEV28,designVoxelwiseEV29,designVoxelwiseEV30,designVoxelwiseEV31,designVoxelwiseEV32,designVoxelwiseEV33,designVoxelwiseEV34,designVoxelwiseEV35,designVoxelwiseEV36'])
    p.wait()
    for f in os.listdir(os.getcwd()):
        if f[-3:] != 'fsf':
            sh.move(f,'/'.join([out_base,f]))
    copes = ['stats/cope' + str(i) + '.nii.gz' for i in range(1,37)] #we want to coregister and warp the copes, the residual and the varcopes
    copes = ['/'.join([out_base,f]) for f in copes]
    dof_file = out_base + '/stats/dof.nii.gz'
    param_estimates = ['stats/pe' + str(i) + '.nii.gz' for i in range(1,37)]
    param_estimates = ['/'.join([out_base,f]) for f in param_estimates]
    residual4d = out_base + '/stats/res4d.nii.gz' #want to coregister, warp, smooth and bandpass this later
    sigmasquareds = out_base + '/stats/sigmasquareds.nii.gz'
    thresholdac = out_base + '/stats/threshac1.nii.gz'
    tstats = ['stats/tstat' + str(i) + '.nii.gz' for i in range(1,37)]
    tstats = ['/'.join([out_base,f]) for f in tstats]
    varcopes = ['stats/varcope' + str(i) + '.nii.gz' for i in range(1,37)]
    varcopes = ['/'.join([out_base,f]) for f in varcopes]
    zstats = ['stats/zstat' + str(i) + '.nii.gz' for i in range(1,37)]
    zstats = ['/'.join([out_base,f]) for f in zstats]
    return copes,dof_file,param_estimates,residual4d,sigmasquareds,thresholdac,tstats,varcopes,zstats

#Define the directories where data is located
parser = ap.ArgumentParser(description='Rest Preprocessing Part 2')
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

rest_data_dir= pardir + "Rest_Preprocessing/Motion_Corrected/_realign0/"
rest_data_file = rest_data_dir + os.listdir(rest_data_dir)[0]
str_data_dir = pardir + "Struct/"
bet_data_dir= pardir + "Rest_Preprocessing/Brain_Extracted/_Brain_Extractor0/"
field_file = pardir + "Rest_Preprocessing/Warp_File_F2MNI/_Concat_Shift_Affine_and_Warp0/"
FSFDIR = pardir + 'FSF/'
try:
    os.mkdir(FSFDIR)
except:
    exceptions.OSError
FSFDIR = FSFDIR + 'Rest/'
try:
    os.mkdir(FSFDIR)
except:
    exceptions.OSError
fsf = open('/gpfs/group/pul8/default/read/Scripts/PNM_ACI.fsf', 'r').read()
parfsf = fsf.replace('PARNUMBER', str(pardir[-4:-1]))
parfsf = parfsf.replace('RESTDATA',rest_data_file)
open("".join([FSFDIR,'design.fsf']), 'w').write(parfsf)
parfsf = "".join([FSFDIR,'design.fsf'])

##----------------------------------------------------------------##
#This section defines the workflow

#Initiate and name Workflow
preproc = Workflow(name = "Rest_Preprocessing_Part_Two")#,base_dir='/media/bschloss/Extra Drive 1/Preprocessing/')

#Put all of the data in the input node
data = Node(ID(fields=['rest','struct','bet','mni','mni_brain','mni_mask',
                       'parfsf', 'field_file'],
               mandatory_inputs=False),name='Data')
data.inputs.rest = [rest_data_file]
data.inputs.struct = [str_data_dir + f for f in os.listdir(str_data_dir) if 'co' in f and 'warp' not in f]
data.inputs.bet = [bet_data_dir + f for f in os.listdir(bet_data_dir)]
data.inputs.mni = '/gpfs/group/pul8/default/read/MNI152_T1_3mm3mm4mm.nii.gz'
data.inputs.mni_brain = '/gpfs/group/pul8/default/read/MNI152_T1_3mm3mm4mm_brain.nii.gz'
data.inputs.mni_mask = '/gpfs/group/pul8/default/read/MNI152_T1_3mm3mm4mm_brain_mask.nii.gz'
data.inputs.parfsf = [parfsf]
data.inputs.field_file = field_file + os.listdir(field_file)[0]

#Use the DataSink function to store all outputs
datasink = Node(nio.DataSink(), name= 'Output')
datasink.inputs.base_directory = pardir +'Rest_Preprocessing/'

#Convert input data to float representation
img2float = MapNode(interface=fsl.ImageMaths(out_data_type='float',
                                                 op_string = '',
                                                 suffix='_dtype'),
                           iterfield=['in_file'],
                           name='img2float')

preproc.connect(data, 'rest', img2float, 'in_file')

#Get mean-image in time direction
meants1 = MapNode(fsl.MeanImage(),
                name= 'Mean_Func1',
                iterfield=['in_file'])
preproc.connect(data,'rest',meants1,'in_file')

#Now we will remove all nuisance signals in native space, before moving the
#data to MNI space.
pnm_feat = MapNode(Function(input_names=['parfsf','rest'],
                            output_names = ['copes','dof_file','param_estimates','residual4d','sigmasquareds','thresholdac','tstats','varcopes','zstats'],
                            function = PNM_Feat),
                            name = "PNM_Feat",
                            iterfield = ['parfsf','rest'])
preproc.connect(data,'rest',pnm_feat,'rest')
preproc.connect(data,'parfsf',pnm_feat,'parfsf')
preproc.connect(pnm_feat,'copes',datasink,'Parametric_Copes')
preproc.connect(pnm_feat,'dof_file',datasink,'Parametric_DOF')
#preproc.connect(pnm_feat,'fstats',datasink,'Parametric_Fstats')
preproc.connect(pnm_feat,'param_estimates',datasink,'Parametric_Param_Estimates')
preproc.connect(pnm_feat,'residual4d',datasink,'Parametric_Residual4D')
preproc.connect(pnm_feat,'sigmasquareds',datasink,'Parametric_Sigma_Squareds')
preproc.connect(pnm_feat,'thresholdac',datasink,'Parametric_AC_Params')
preproc.connect(pnm_feat,'tstats',datasink,'Parametric_Tstats')
preproc.connect(pnm_feat,'varcopes',datasink,'Parametric_Varcopes')
#preproc.connect(pnm_feat,'zfstats',datasink,'Parametric_Z_Fstats')
preproc.connect(pnm_feat,'zstats',datasink,'Parametric_Z_Tstats')

#Add the mean back in
add_mean_back_in1 = MapNode(fsl.BinaryMaths(operation='add'),
                   name = 'Add_Mean_Back_In1',
                   iterfield = ['in_file','operand_file'])
preproc.connect(pnm_feat,'residual4d',add_mean_back_in1,'in_file')
preproc.connect(meants1,('out_file',unnest_list),add_mean_back_in1,'operand_file')
preproc.connect(add_mean_back_in1,'out_file',datasink,'Filtered_Rest')

#Apply warp to functional data that has been registered to structural image
#converting it to MNI space using Nonlinear registartion
#warps
apply_warp = MapNode(fsl.ApplyWarp(),name='Filter_Warper_F2MNI',
                     iterfield=['in_file'])
preproc.connect(data,'mni',apply_warp,'ref_file')
preproc.connect(data,'field_file',apply_warp,'field_file')
preproc.connect(data,'mni_mask',apply_warp,'mask_file')
preproc.connect(add_mean_back_in1,'out_file',apply_warp,'in_file')
preproc.connect(apply_warp,'out_file',datasink,'Filtered_F2MNI_NL_Warped')

#Apply warp to functional data that has been registered to structural image
#converting it to MNI space using Nonlinear registartion
#warps
apply_warp_copes = MapNode(fsl.ApplyWarp(),name='Copes_Warper_F2MNI',
                     iterfield=['in_file'])
preproc.connect(data,'mni',apply_warp_copes,'ref_file')
preproc.connect(data,'field_file',apply_warp_copes,'field_file')
preproc.connect(data,'mni_mask',apply_warp_copes,'mask_file')
preproc.connect(pnm_feat,('copes',unnest_list),apply_warp_copes,'in_file')
preproc.connect(apply_warp_copes,'out_file',datasink,'Copes_F2MNI_NL_Warped')

#Apply warp to functional data that has been registered to structural image
#converting it to MNI space using Nonlinear registartion
#warps
apply_warp_varcopes = MapNode(fsl.ApplyWarp(),name='Varcopes_Warper_F2MNI',
                     iterfield=['in_file'])
preproc.connect(data,'mni',apply_warp_varcopes,'ref_file')
preproc.connect(data,'field_file',apply_warp_varcopes,'field_file')
preproc.connect(data,'mni_mask',apply_warp_varcopes,'mask_file')
preproc.connect(pnm_feat,('varcopes',unnest_list),apply_warp_varcopes,'in_file')
preproc.connect(apply_warp_varcopes,'out_file',datasink,'Varcopes_F2MNI_NL_Warped')

#Get mean-image in time direction
meants2 = MapNode(fsl.MeanImage(),
                name= 'Mean_Func2',
                iterfield=['in_file'])
preproc.connect(apply_warp,'out_file',meants2,'in_file')

#Bandpass Filter the data
bandpass = MapNode(fsl.TemporalFilter(lowpass_sigma = float((10.0/2.0)/2.355),
                                      highpass_sigma = float((111.11111111111/2.0)/2.355)),
                   name='Bandpass',
                   iterfield=['in_file'])
preproc.connect(apply_warp,'out_file',bandpass,'in_file')

#Add the mean back in
add_mean_back_in2 = MapNode(fsl.BinaryMaths(operation='add'),
                   name = 'Add_Mean_Back_In2',
                   iterfield = ['in_file','operand_file'])
preproc.connect(bandpass,'out_file',add_mean_back_in2,'in_file')
preproc.connect(meants2,'out_file',add_mean_back_in2,'operand_file')
preproc.connect(add_mean_back_in2,'out_file',datasink,'Bandpass_Rest')

#Spatially smooth functional data in MNI space with 8mm FWHM Gaussian
smoother = MapNode(interface=fsl.IsotropicSmooth(fwhm=8),
                      name='Smoother',
                      iterfield=['in_file'])
preproc.connect(add_mean_back_in2, 'out_file', smoother, 'in_file')
preproc.connect(smoother, 'out_file', datasink, 'Smoothed')

preproc.run(plugin='MultiProc', plugin_args={'n_procs' : 3})
