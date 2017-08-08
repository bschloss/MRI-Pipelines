# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:11:21 2015

@author: bschloss
"""

import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
from nipype.interfaces.utility import IdentityInterface as ID
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import Function
import nibabel as nib
import os
import argparse as ap

parser = ap.ArgumentParser(description='Preprocess DTI data and put in new folder')
parser.add_argument('pardir', metavar="ParDir", type=str,
                    help="Path to participant's directory")                                                  
args = parser.parse_args()
args = parser.parse_args()
pardir= '/gpfs/group/pul8/default/read/'
if len(args.pardir) == 1:
    pardir += '00' + args.pardir + '/'
elif len(args.pardir) == 2:
    pardir += '0' + args.pardir + '/'
else:
    pardir += args.pardir + '/'

#Get list of subjects to be analyzed at T1 and DTI
struct_dir = pardir + 'Struct/'
struct = [struct_dir + f for f in os.listdir(struct_dir) if 'co' in f and 'warp' not in f]

dwi_dir = pardir + 'DTI/'
dwi = [f for f in 
            [f for f in 
                    [dwi_dir + f for f in os.listdir(dwi_dir) 
                    if 'nii.gz' in f] 
            if len(nib.load(f).shape)==4] 
      if nib.load(f).shape[3] == 70]      
b0PA = [dwi_dir + f for f in os.listdir(dwi_dir) if 'PA' in f][0]
bvecs = [dwi_dir + f for f in os.listdir(dwi_dir) if 'bvec' in f]
bvals = [dwi_dir + f for f in os.listdir(dwi_dir) if 'bval' in f]

##-----------------------------------------------------------------##
##This section defines functions that will be used during prerpocessing but
## will not act as actual "nodes" in the preprocessing pipeline    
def get_crash(crashdir):
    import os
    import gzip as gz
    import pickle as pkl
    crash = [crashdir + f for f in os.listdir(crashdir) if 'crash' in f]
    return [pkl.load(gz.open(f,'rb')) for f in crash]

def topup_merge_list(b0AP,b0PA):

    if type(b0AP )== list:
        new = []
        for i in range(len(b0AP)):
            new.append([b0AP[i],b0PA])
    if type(b0AP) == str:
        new = [b0AP,b0PA]
    return new

def topup_prefix(fieldcoef):
    if type(fieldcoef) == list:
        return [f.replace('_fieldcoef.nii.gz','') for f in fieldcoef]
    if type(fieldcoef) == str:
        return fieldcoef.replace('_fieldcoef.nii.gz','')
        
def bedpostx_gpu(bvals,bvecs,dwi,mask,base_dir):
    
    import shutil
    import subprocess as sp
    import os
    
    indir = base_dir.rstrip('/') + '/BEDPOSTX/_BEDPOSTX_GPU0/'
    os.makedirs(indir)
    
    shutil.copy(bvals,''.join([indir,'bvals']))
    shutil.copy(bvecs,''.join([indir,'bvecs']))
    shutil.copy(dwi,''.join([indir,'data.nii.gz']))
    shutil.copy(mask,''.join([indir,'nodif_brain_mask.nii.gz']))
    
    bdp_gpu = sp.Popen(['bedpostx_gpu',indir])
    sp.Popen.wait(bdp_gpu)
    
    outputdir = indir.rstrip('/') + '.bedpostX/'
    merged_fsamples = [outputdir + 'merged_f' + str(i) + 'samples.nii.gz' for i in range(1,4)]
    merged_phsamples = [outputdir + 'merged_ph' + str(i) + 'samples.nii.gz' for i in range(1,4)]
    merged_thsamples = [outputdir + 'merged_th' + str(i) + 'samples.nii.gz' for i in range(1,4)]
    
    return merged_fsamples, merged_phsamples, merged_thsamples

def eddy_cuda(dwi,mask,index,acqp,bvecs,bvals,topup):
    import subprocess as sp
    
    eddy_cuda = sp.Popen(['eddy_cuda',
                          ''.join(['--imain=',dwi]),
                          ''.join(['--mask=',mask]),
                          ''.join(['--acqp=',acqp]),
                          ''.join(['--index=',index]),
                          ''.join(['--bvecs=',bvecs]),
                          ''.join(['--bvals=',bvals]),
                          ''.join(['--topup=',topup]),
                          ''.join(['--out=','eddy_corrected'])])
    sp.Popen.wait(eddy_cuda)
    
    return None

def eddy_openmp(dwi,mask,index,acqp,bvecs,bvals,topup):
    import subprocess as sp
    import os
    outbase = os.getcwd() 
    eddy_openmp = sp.Popen(['eddy_openmp',
                          ''.join(['--imain=',dwi]),
                          ''.join(['--mask=',mask]),
                          ''.join(['--acqp=',acqp]),
                          ''.join(['--index=',index]),
                          ''.join(['--bvecs=',bvecs]),
                          ''.join(['--bvals=',bvals]),
                          ''.join(['--topup=',topup]),
                          ''.join(['--out=','eddy_corrected'])])
    sp.Popen.wait(eddy_openmp)

    return '/'.join([outbase,'eddy_corrected'])
    
def choose_wm(fs):
        return [fs[0][2]]    
        
##----------------------------------------------------------------##
#This section defines the workflow

#Define Workflow
preproc = Workflow(name = "Preprocessing")#,base_dir='/media/bschloss/Extra Drive 1/Preprocessing/')

#Use DataGrabber function to get all of the data
data = Node(ID(fields=['dwi','struct','mni','mni_brain','mni_mask','bvecs',
                       'bvals','bedpostx_dir','datain','b0PA','index'],
	    mandatory_inputs=False),
	    name='Data')
data.inputs.dwi = dwi
data.inputs.b0PA = b0PA
data.inputs.struct = struct
data.inputs.mni = '/gpfs/group/pul8/default/sw/fsl/5.0.10/data/standard/MNI152_T1_2mm.nii.gz'
data.inputs.mni_brain = '/gpfs/group/pul8/default/sw/fsl/5.0.10/data/standard/MNI152_T1_2mm_brain.nii.gz'
data.inputs.mni_mask = '/gpfs/group/pul8/default/sw/fsl/5.0.10/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
data.inputs.bvecs = bvecs
data.inputs.bvals = bvals
data.inputs.datain = '/gpfs/group/pul8/default/read/datain_dti.txt'
data.inputs.index = '/gpfs/group/pul8/default/read/index.txt'
data.inputs.bedpostx_dir = pardir + 'DTI_Preprocessing/'

#Use the DataSink function to store all outputs
datasink = Node(nio.DataSink(), name= 'Ouput')
datasink.inputs.base_directory = pardir + 'DTI_Preprocessing/'

#Convert input data to float representation
img2float = MapNode(interface=fsl.ImageMaths(out_data_type='float',
                                                 op_string = '',
                                                 suffix='_dtype'),
                           iterfield=['in_file'],
                           name='img2float')
                           
preproc.connect(data, 'dwi', img2float, 'in_file')

#Extract first b0 volume
b0AP = MapNode(interface = fsl.ExtractROI(t_min=0,t_size=1),
               name = 'B0AP',
               iterfield=['in_file'])
preproc.connect(img2float,'out_file',b0AP,'in_file')
preproc.connect(b0AP,'roi_file',datasink,'B0_AP')

#Merge B0AP with B0PA in lists
b0APPA = MapNode(Function(input_names = ['b0AP','b0PA'],
                        output_names = ['merged_b0appa'],
                        function = topup_merge_list),
                 name = 'Topup_Merge_List',
                 iterfield = ['b0AP'])
preproc.connect(b0AP,'roi_file',b0APPA,'b0AP')
preproc.connect(data,'b0PA',b0APPA,'b0PA')

#Merge B0AP with B0PA into a single image
appa_merger = MapNode(interface=fsl.Merge(dimension='t'),
                          iterfield=['in_files'],
                          name="APPA_Merger")
preproc.connect(b0APPA,'merged_b0appa',appa_merger,'in_files')
preproc.connect(appa_merger,'merged_file',datasink,'B0_AP_PA_Merged')

#Run Top UP
topup = MapNode(fsl.epi.TOPUP(),
                name = 'TOPUP',
                iterfield = ['in_file'])
preproc.connect(data,'datain',topup,'encoding_file')
preproc.connect(appa_merger,'merged_file',topup,'in_file')
preproc.connect(topup,'out_corrected',datasink,'Phase_Unwarped_B0')
preproc.connect(topup,'out_enc_file',datasink,'Apply_TOPUP_Encoding_File')
preproc.connect(topup,'out_field',datasink,'Fieldmap_Field')
#preproc.connect(topup,'out_movpar',datasink,'Movement_Pars')

#Mean B0
mean_b0 = MapNode(fsl.MeanImage(),
                  name='Mean_B0',
                  iterfield = 'in_file')
preproc.connect(topup,'out_corrected',mean_b0,'in_file')
preproc.connect(mean_b0,'out_file',datasink,'Mean_B0')

#Brain Extract Mean B0
bet_b0 = MapNode(fsl.BET(mask=True,robust=True),
                 name='B0_Extractor',
                 iterfield= ['in_file'])
preproc.connect(mean_b0,'out_file',bet_b0,'in_file')
preproc.connect(bet_b0,'out_file',datasink,'Brain_Extracted_B0')
preproc.connect(bet_b0,'mask_file',datasink,'B0_Brain_Mask')


'''
#Use eddy_cuda to motion correct and apply distortion correction
eddy = MapNode(Function(input_names = ['dwi','mask','index','acqp','bvecs','bvals','topup'],
                        output_names = ['merged_fsamples','merged_phsamples','merged_thsamples'],
                        function = eddy_cuda),
                name = 'Eddy_Correct',
                iterfield=['dwi','topup'])
preproc.connect(img2float,'out_file',eddy,'dwi')
preproc.connect(bet_b0,'mask_file',eddy,'mask')
preproc.connect(data,'index',eddy,'index')
preproc.connect(data,'datain',eddy,'acqp')
preproc.connect(data,'bvecs',eddy,'bvecs')
preproc.connect(data,'bvals',eddy,'bvals')
preproc.connect(topup,('out_fieldcoef',topup_prefix),eddy,'topup')
preproc.connect(eddy,'eddy_corrected',datasink,'Eddy_Cuda')
'''

eddy = MapNode(fsl.epi.Eddy(flm = 'quadratic',
                            num_threads=1),
               name = 'Eddy',
               iterfield = ['in_file','in_mask',
                            'in_topup_fieldcoef',
                            'in_topup_movpar',
                            'in_bvec','in_bval'])
preproc.connect(img2float,'out_file',eddy,'in_file')
preproc.connect(bet_b0,'mask_file',eddy,'in_mask')
preproc.connect(topup,'out_fieldcoef',eddy,'in_topup_fieldcoef')
preproc.connect(data,'datain',eddy,'in_acqp')
preproc.connect(data,'bvals',eddy,'in_bval')
preproc.connect(data,'bvecs',eddy,'in_bvec')
preproc.connect(data,'index',eddy,'in_index')
preproc.connect(topup,'out_movpar',eddy,'in_topup_movpar')
preproc.connect(eddy,'out_corrected',datasink,'Eddy')

#Get mean image of motion corrected data
fslroi = MapNode(interface=fsl.ExtractROI(t_min=0,t_size=1),
                 name='fslroi',
                 iterfield = ['in_file'])
preproc.connect(img2float,'out_file',fslroi,'in_file')
preproc.connect(fslroi,'roi_file',datasink,'Nodif_Image')

#Skull strip the mean functional image
nodif_strip = MapNode(fsl.BET(mask=True, frac=.30),
                      name='Nodif_Strip', 
                      iterfield=['in_file'])
preproc.connect(fslroi,'roi_file',nodif_strip, 'in_file')
preproc.connect(nodif_strip,'out_file', datasink,'Nodif_Stripped_Image')
preproc.connect(nodif_strip,'mask_file',datasink,'Nodif_Brain_Mask')

dtifit = MapNode(interface=fsl.DTIFit(save_tensor=True),
                 name='DTIFIT',
                 iterfield = ['bvecs','bvals','dwi','mask'])
preproc.connect(data,'bvecs',dtifit,'bvecs')
preproc.connect(data,'bvals',dtifit,'bvals')
preproc.connect(eddy,'out_corrected',dtifit,'dwi')
preproc.connect(nodif_strip,'mask_file',dtifit,'mask')
preproc.connect(dtifit,'FA',datasink,'DTIFIT_FA')
preproc.connect(dtifit,'L1',datasink,'DTIFIT_L1')
preproc.connect(dtifit,'L2',datasink,'DTIFIT_L2')
preproc.connect(dtifit,'L3',datasink,'DTIFIT_L3')
preproc.connect(dtifit,'MD',datasink,'DTIFIT_MD')
preproc.connect(dtifit,'MO',datasink,'DTIFIT_MO')
preproc.connect(dtifit,'S0',datasink,'DTIFIT_S0')
preproc.connect(dtifit,'V1',datasink,'DTIFIT_V1')
preproc.connect(dtifit,'V2',datasink,'DTIFIT_V2')
preproc.connect(dtifit,'V3',datasink,'DTIFIT_V3')
preproc.connect(dtifit,'tensor',datasink,'DTIFIT_Tensor')

'''
bedpostx = MapNode(interface=Function(input_names = ['bvals','bvecs','dwi','mask','base_dir'],
                                                                output_names = ['merged_fsamples','merged_phsamples','merged_thsamples'],
                                                                function = bedpostx_gpu),
                   name='BEDPOSTX',
                   iterfield = ['bvals','bvecs','dwi','mask'])
preproc.connect(data,'bedpostx_dir',bedpostx,'base_dir')
preproc.connect(data,'bvecs',bedpostx,'bvecs')
preproc.connect(data,'bvals',bedpostx,'bvals')
preproc.connect(eddy,'out_corrected',bedpostx,'dwi')
preproc.connect(nodif_strip,'mask_file',bedpostx,'mask')
preproc.connect(bedpostx,'merged_fsamples',datasink,'BEDPOSTX_Merged_Fsamples')
preproc.connect(bedpostx,'merged_phsamples',datasink,'BEDPOSTX_Merged_PHsamples')
preproc.connect(bedpostx,'merged_thsamples',datasink,'BEDPOSTX_Merged_THsamples')
'''

bedpostx = MapNode(fsl.BEDPOSTX(n_fibres=3),
                   name='BEDPOSTX',
                   iterfield = ['bvals','bvecs','dwi','mask'])
preproc.connect(data,'bvecs',bedpostx,'bvecs')
preproc.connect(data,'bvals',bedpostx,'bvals')
preproc.connect(eddy,'out_corrected',bedpostx,'dwi')
preproc.connect(nodif_strip,'mask_file',bedpostx,'mask')
preproc.connect(bedpostx,'merged_fsamples',datasink,'BEDPOSTX_Merged_Fsamples')
preproc.connect(bedpostx,'merged_phsamples',datasink,'BEDPOSTX_Merged_PHsamples')
preproc.connect(bedpostx,'merged_thsamples',datasink,'BEDPOSTX_Merged_THsamples')

#Brain Extract structural images
bet = MapNode(fsl.BET(robust=True),
              name='Brain_Extractor',
              iterfield=['in_file'])
preproc.connect(data,'struct',bet,'in_file')
preproc.connect(bet,'out_file',datasink,'Brain_Extracted')


#Segment the white matter
fast = MapNode(fsl.FAST(img_type=1,segments = True,no_pve =True),
               name = 'Segmenter',
               iterfield = ['in_files'])

preproc.connect(bet,'out_file',fast,'in_files')
preproc.connect(fast,('tissue_class_files',choose_wm),datasink,'WM_Seg')
    
#Coregister the mean functional data image to the respective structural image
register_d2s = MapNode(fsl.EpiReg(),
                       name='Register_D2S',
                       iterfield=['epi','t1_head','t1_brain','wmseg'])
preproc.connect(nodif_strip,'out_file',register_d2s,'epi')
preproc.connect(data,'struct',register_d2s,'t1_head')
preproc.connect(bet,'out_file',register_d2s,'t1_brain')
preproc.connect(fast,('tissue_class_files',choose_wm),register_d2s,'wmseg')
preproc.connect(register_d2s,'out_file',datasink,'Registered_Mean_F2S')
preproc.connect(register_d2s,'epi2str_mat',datasink,'Affine_D2S')
preproc.connect(register_d2s,'wmedge',datasink,'WM_Edges')

#Now we register the functional data to MNI space
#Register the structural data to MNI using FLIRT
register_s2MNI_L = MapNode(fsl.FLIRT(dof=12),
                           name='Registration_s2MNI_L',
                           iterfield=['in_file'])
preproc.connect(bet,'out_file',register_s2MNI_L,'in_file')
preproc.connect(data,'mni_brain',register_s2MNI_L,'reference')
preproc.connect(register_s2MNI_L,'out_matrix_file',datasink,'Affine_s2MNI')
preproc.connect(register_s2MNI_L,'out_file',datasink,'Registered_S2MNI_L')

#Register the structural data to MNI using FNIRT
register_s2MNI_NL = MapNode(fsl.FNIRT(field_file=True,
                                      fieldcoeff_file=True,
                                      skip_inmask = True,
                                      apply_refmask = [0,0,0,1,1,1],
                                      regularization_lambda=[400,200,150,75,60,45],
                                      subsampling_scheme = [8,4,2,2,1,1],
                                      max_nonlin_iter = [5,5,5,10,5,10],
                                      in_fwhm = [8,6,4,2,2,2],
                                      ref_fwhm = [6,4,2,0,0,0],
                                      apply_intensity_mapping = [1,1,1,1,1,0]),
                            name='Registration_s2MNI_NL',
                            iterfield=['in_file','affine_file'])
preproc.connect(data,'struct',register_s2MNI_NL,'in_file')
preproc.connect(register_s2MNI_L,'out_matrix_file',register_s2MNI_NL,'affine_file')
preproc.connect(data,'mni',register_s2MNI_NL,'ref_file')
preproc.connect(data,'mni_mask',register_s2MNI_NL,'refmask_file')
preproc.connect(register_s2MNI_NL,'field_file',datasink,'Warp_Field_File')
preproc.connect(register_s2MNI_NL,'fieldcoeff_file',datasink,'Warp_Field_Coefficient_File')
preproc.connect(register_s2MNI_NL,'warped_file',datasink,'Registered_s2MNI_NL_Warped')

warp_D2MNI = MapNode(fsl.ConvertWarp(relwarp=True),
                     name = "D2MNI_Concatenated_Warp",
                     iterfield = ['reference','warp1','premat'])
preproc.connect(data,'mni',warp_D2MNI,'reference')
preproc.connect(register_s2MNI_NL,'fieldcoeff_file',warp_D2MNI,'warp1')
preproc.connect(register_d2s,'epi2str_mat',warp_D2MNI,'premat')
preproc.connect(warp_D2MNI,'out_file',datasink,'D2MNI_Warp_Field_File')

#Apply warp to functional data that has been registered to structural image
#converting it to MNI space using Nonlinear registartion
apply_warp = MapNode(fsl.ApplyWarp(relwarp=True),
                     name='Warper_D2MNI',
                     iterfield=['in_file','field_file'])
preproc.connect(data,'mni',apply_warp,'ref_file')
preproc.connect(warp_D2MNI,'out_file',apply_warp,'field_file')
preproc.connect(data,'mni_mask',apply_warp,'mask_file')
preproc.connect(nodif_strip,'out_file',apply_warp,'in_file')
preproc.connect(apply_warp,'out_file',datasink,'Registered_D2MNI_NL_Warped')  

affine_s2d = MapNode(fsl.ConvertXFM(invert_xfm=True),
                     name = "Affine_S2D_Inverted",
                     iterfield = ['in_file'])
preproc.connect(register_d2s,'epi2str_mat',affine_s2d,'in_file')
preproc.connect(affine_s2d,'out_file',datasink,'Affine_S2D')

warp_MNI2S = MapNode(fsl.InvWarp(relative=True),
                     name = "MNI2S_Inverted_Warp",
                     iterfield = ['warp','reference'])
preproc.connect(register_s2MNI_NL,'fieldcoeff_file',warp_MNI2S,'warp')
preproc.connect(data,'struct',warp_MNI2S,'reference',)
preproc.connect(warp_MNI2S,'inverse_warp',datasink,'MNI2S_Warp_Field_File')

warp_MNI2D = MapNode(fsl.ConvertWarp(relwarp=True),
                     name = "MNI2D_Concatenated_Warp",
                     iterfield = ['reference','warp1','postmat'])
preproc.connect(nodif_strip,'out_file',warp_MNI2D,'reference')
preproc.connect(warp_MNI2S, 'inverse_warp',warp_MNI2D,'warp1')
preproc.connect(affine_s2d,'out_file',warp_MNI2D,'postmat')
preproc.connect(warp_MNI2D,'out_file',datasink,'MNI2D_Warp_Field_File')

#preproc.write_graph(dotfilename='DTI_Preprocessing_Graph',format='svg')
#preproc.write_graph(dotfilename='DTI_Preprocessing_Graph',format='svg',graph2use='exec')
preproc.run(plugin='MultiProc',plugin_args={'n_procs' : 5})
