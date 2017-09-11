# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:36:21 2017

@author: bschloss
"""

import os
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
from nipype.interfaces.utility import Function
from nipype.interfaces.utility import IdentityInterface as ID
from nipype.pipeline.engine import Workflow, Node, MapNode
from math import pi as pi

##-----------------------------------------------------------------##
##This section defines functions that will be used during prerpocessing but
## will not act as actual "nodes" in the preprocessing pipeline    
def filter_design_mat(csts,wmts,td):
    import os
    cs = [num.replace(' \n','\t') for num in open(csts,'r').readlines()]
    wm = [num.replace(' \n','\t') for num in open(wmts,'r').readlines()]
    td = open(td,'r').readlines()[5:254]
    design = [cs[i] + wm[i] + td[i] for i in range(249)]
    
    file_name = csts.split('/')[len(csts.split('/'))-1].replace('dtype_despike_mcf_masked_flirt_ts.txt','filter_design.mat') 
    txt = ''
    for line in design:
        txt += line
                
    with open(file_name,'w') as f:
        f.write(txt)
    return os.path.abspath(file_name)
    
def list_of_lists(in_file):
    return [[item] for item in in_file]

def topup_merge_list(seAP,sePA):

    if type(seAP) == list:
        new = []
        for i in range(len(seAP)):
            print seAP[i],sePA[i]
            new.append([seAP[i],sePA[i]])
    elif type(seAP) == str:
        new = [seAP,sePA]
    return new
    
def run_merger(in_files):
    new_list = []
    for i in range(len(list)):
        if i%2==0:
            new_list.append([in_files[i],in_files[i+1]])
    return new_list
    
def choose_wm(fs):
    return [fs[0][2]]

def choose_csf(fs):
    return [fs[0][0]]
    
def choose_wm5(fs):
    return [fs[0][2] for i in range(5)]

def choose_csf5(fs):
    return [fs[0][0] for i in range(5)]

#Create EV file
def wm_ev(wmts):
    import os
    import numpy.random as random
    import string
    import exceptions
    out_base = ''
    while out_base == '':
        letters = string.ascii_letters
        out_base = ''.join(['/tmp/tmp',''.join([letters[i] for i in random.choice(len(letters),10)])])
        out_base += '/WM_EV/'
        try:
            os.makedirs(out_base)
        except:
            exceptions.OSError
            out_base = ''
    outf = out_base + 'WM.run001.txt'
    txt = ''
    secs = 0
    for val in open(wmts,'r').read().replace(' ','').rstrip('\n').split('\n'):
        txt += '\t'.join([str(secs),str(0),val.rstrip('n')]) +'\n'
        secs += 2
    open(outf,'w').write(txt.rstrip('\n'))
    return outf
    
#Create EV file
def csf_ev(csfts):
    import os
    import numpy.random as random
    import string
    import exceptions
    out_base = ''
    while out_base == '':
        letters = string.ascii_letters
        out_base = ''.join(['/tmp/tmp',''.join([letters[i] for i in random.choice(len(letters),10)])])
        out_base += '/CSF_EV/'
        try:
            os.makedirs(out_base)
        except:
            exceptions.OSError
            out_base = ''
    outf = out_base + 'CSF.run001.txt'
    txt = ''
    secs = 0
    for val in open(csfts,'r').read().replace(' ','').rstrip('\n').split('\n'):
        txt += '\t'.join([str(secs),str(0),val.rstrip('n')]) +'\n'
        secs += 2
    open(outf,'w').write(txt.rstrip('\n'))
    return outf
    
def repfiles(fs):
    new = []
    for f in fs:
        for i in range(5):
            new.append(f)
    return new
def combine_covariates(mp,csf,wm):
    import os
    import numpy.random as random
    import string
    import exceptions
    out_base = ''
    while out_base == '':
        letters = string.ascii_letters
        out_base = ''.join(['/tmp/tmp',''.join([letters[i] for i in random.choice(len(letters),10)])])
        out_base += '/MP_CSF_WM_Covariates/'
        try:
            os.makedirs(out_base)
        except:
            exceptions.OSError
            out_base = ''
    motion = open(mp,'r').readlines()
    sf = open(csf,'r').readlines()
    white = open(wm,'r').readlines()
    out = ''
    for i in range(len(motion)):
        line = motion[i].split() + sf[i].split() + white[i].split()
        out += '  '.join(line) + '  \n'
    out.rstrip('\n')
    covariates = out_base + 'mp_csf_wm_cov' + mp.split('/')[-2][-1] + '.par'
    open(covariates,'w').write(out)
    return covariates

##----------------------------------------------------------------##
#This section defines the workflow

#Define Workflow
preproc = Workflow(name = "fMRI_Preprocessing")#,base_dir='/media/bschloss/Extra Drive 1/Preprocessing/')

#Use DataGrabber function to get all of the data
data = Node(ID(fields=['func','struct','mni','mni_brain','mni_mask','mag','phase','seAP','sePA','datain'],
               mandatory_inputs=False),name='Data')
data.inputs.func = ['/gpfs/group/pul8/default/read/PILOT/Func/Run1/MBTR400EF6Run1s003a001.nii.gz']
data.inputs.struct = ['/gpfs/group/pul8/default/read/PILOT/Struct/cot1mpragesagp2isos010a1001.nii.gz']
data.inputs.mni = '/gpfs/group/pul8/default/read/MNI152_T1_3mm3mm4mm.nii.gz'
data.inputs.mni_brain = '/gpfs/group/pul8/default/read/MNI152_T1_3mm3mm4mm_brain.nii.gz'
data.inputs.mni_mask = '/gpfs/group/pul8/default/read/MNI152_T1_3mm3mm4mm_brain_mask.nii.gz'
#data.inputs.mag = [fieldmap_dir + f for f in os.listdir(fieldmap_dir) if '1001.nii.gz' in f]
#data.inputs.phase = [fieldmap_dir + f for f in os.listdir(fieldmap_dir) if '2001.nii.gz' in f]
data.inputs.seAP = ['/gpfs/group/pul8/default/read/PILOT/Fieldmap/cmrrmbep2dse4AP30s004a001.nii.gz']
data.inputs.sePA = ['/gpfs/group/pul8/default/read/PILOT/Fieldmap/cmrrmbep2dse4PA30s005a001.nii.gz']
data.inputs.datain = '/gpfs/group/pul8/default/read/datain_fMRI.txt'
#Use the DataSink function to store all outputs
datasink = Node(nio.DataSink(), name= 'Output')
datasink.inputs.base_directory = '/gpfs/group/pul8/default/read/PILOT/fMRI_Preprocessing/'

#Convert input data to float representation
img2float = MapNode(interface=fsl.ImageMaths(out_data_type='float',
                                                 op_string = '',
                                                 suffix='_dtype'),
                           iterfield=['in_file'],
                           name='img2float')
                           
preproc.connect(data, 'func', img2float, 'in_file')

AP0 = MapNode(fsl.ExtractROI(t_min=0,t_size=1),
                  name = 'AP0',
                  iterfield = ['in_file'])
preproc.connect(data,'seAP',AP0,'in_file')
preproc.connect(AP0,'roi_file',datasink,'AP0')

PA0 = MapNode(fsl.ExtractROI(t_min=0,t_size=1),
                  name = 'PA0',
                  iterfield = ['in_file'])
preproc.connect(data,'sePA',PA0,'in_file')
preproc.connect(PA0,'roi_file',datasink,'Mean_PA')

#Merge B0AP with B0PA in lists
seAPPA = MapNode(Function(input_names = ['seAP','sePA'],
                        output_names = ['merged_seappa'],
                        function = topup_merge_list),
                 name = 'Topup_Merge_List',
                 iterfield = ['seAP','sePA'])
preproc.connect(AP0,'roi_file',seAPPA,'seAP')
preproc.connect(PA0,'roi_file',seAPPA,'sePA')

#Merge B0AP with B0PA into a single image
appa_merger = MapNode(interface=fsl.Merge(dimension='t'),
                          iterfield=['in_files'],
                          name="APPA_Merger")
preproc.connect(seAPPA,'merged_seappa',appa_merger,'in_files')
preproc.connect(appa_merger,'merged_file',datasink,'SE_AP_PA_Merged')

#Run Top UP
topup = MapNode(fsl.epi.TOPUP(),
                name = 'TOPUP',
                iterfield = ['in_file'])
preproc.connect(data,'datain',topup,'encoding_file')
preproc.connect(appa_merger,'merged_file',topup,'in_file')
preproc.connect(topup,'out_enc_file',datasink,'Apply_TOPUP_Encoding_File')

#Convert Fieldmap to radians
fmap2rads = MapNode(fsl.BinaryMaths(operation='mul',operand_value=2*pi),
                    name = 'Fieldmap_to_Radians',
                    iterfield = ['in_file'])
preproc.connect(topup,'out_field',fmap2rads,'in_file')
preproc.connect(fmap2rads,'out_file',datasink,'Fieldmap_Radians')

#Estimate signal loss
se_sig_loss = MapNode(interface=fsl.epi.SigLoss(echo_time = .0512,
                                                slice_direction = 'z'),
                       name = 'Estimate_Signal_Loss',
                       iterfield = ['in_file'])
preproc.connect(fmap2rads,'out_file',se_sig_loss,'in_file')
preproc.connect(se_sig_loss,'out_file',datasink,'Estimated_Signal_Loss')

#Average Magnitude
se_mag_mean = MapNode(fsl.MeanImage(),
                  name = 'Magnitude_Image',
                  iterfield = ['in_file'])
preproc.connect(topup,'out_corrected',se_mag_mean,'in_file')
preproc.connect(se_mag_mean,'out_file',datasink,'Magnitude_Image')

#Brain Extrac Magnitude Image
se_mag_bet = MapNode(fsl.BET(robust=True,mask=True),
                  name = 'BET_Magnitude',
                  iterfield = ['in_file'])
preproc.connect(se_mag_mean,'out_file',se_mag_bet,'in_file')
preproc.connect(se_mag_bet,'out_file',datasink,'Brain_Extracted_SE_Mag')

#Brain Extract structural images
bet = MapNode(fsl.BET(robust=True),name='Brain_Extractor',iterfield=['in_file'])
preproc.connect(data,'struct',bet,'in_file')
preproc.connect(bet,'out_file',datasink,'Brain_Extracted')

#Segment the white matter
fast = MapNode(fsl.FAST(img_type=1,segments = True,no_pve =True),
               name = 'Segmenter',
               iterfield = ['in_files'])

preproc.connect(bet,'out_file',fast,'in_files')
preproc.connect(fast,('tissue_class_files',choose_wm),datasink,'WM_Seg')

#Motion correct the functional data                     
motion_correct = MapNode(interface=fsl.MCFLIRT(save_mats = True,
                                           save_plots = True,
                                           stats_imgs = True,
                                           mean_vol = True),
                        name='realign',
                        iterfield = ['in_file'])
preproc.connect(img2float, 'out_file', motion_correct,'in_file')
preproc.connect(motion_correct,'mean_img',datasink,'Mean_Func')
preproc.connect(motion_correct, 'out_file',datasink,'Motion_Corrected')
preproc.connect(motion_correct, 'par_file', datasink,'MotionPars')

#Plot the motion parameters
plot_motion = MapNode(interface=fsl.PlotMotionParams(in_source='fsl',
                                                     plot_type='displacement'),
                            name='plot_motion',
                            iterfield=['in_file'])
preproc.connect(motion_correct, 'par_file', plot_motion, 'in_file')
preproc.connect(plot_motion, 'out_file', datasink, 'Motion_Plots')

#Skull strip the mean functional image
mean_strip = MapNode(fsl.BET(robust=True,mask=True),
                     name='Mean_Strip', 
                     iterfield=['in_file'])
preproc.connect(motion_correct,'mean_img',mean_strip, 'in_file')
preproc.connect(mean_strip,'out_file', datasink,'Mean_Stripped_Image')

#Apply the mask from mean brain extracted func image
functional_brain_mask = MapNode(fsl.ApplyMask(),
                                name='Functional_Brain_Extracted',
                                iterfield=['in_file','mask_file'])
preproc.connect(mean_strip,'mask_file',functional_brain_mask,'mask_file')
preproc.connect(motion_correct,'out_file',functional_brain_mask,'in_file')
preproc.connect(functional_brain_mask,'out_file', datasink, 'Skull_Stripped_Func')

#Coregister the mean functional data image to the respective structural image
#Coregister the mean functional data image to the respective structural image
register_f2s = MapNode(fsl.EpiReg(pedir='-y',echospacing = .00058),
                       name='Register_F2S',
                       iterfield=['epi','t1_brain','t1_head','fmap','fmapmag','fmapmagbrain','wmseg'])
preproc.connect(functional_brain_mask,'out_file',register_f2s,'epi')
preproc.connect(bet,'out_file',register_f2s,'t1_brain')
preproc.connect(data,'struct',register_f2s,'t1_head')
preproc.connect(fmap2rads,'out_file',register_f2s,'fmap')
preproc.connect(se_mag_mean,'out_file',register_f2s,'fmapmag')
preproc.connect(se_mag_bet,'out_file',register_f2s,'fmapmagbrain')
preproc.connect(fast,('tissue_class_files',choose_wm),register_f2s,'wmseg')
preproc.connect(register_f2s,'epi2str_mat',datasink,'F2S_Affine')
preproc.connect(register_f2s,'out_file',datasink,'Coregistered_F2S')
preproc.connect(register_f2s,'wmedge',datasink,'WM_Edges')
preproc.connect(register_f2s,'shiftmap',datasink,'Shift_Map')                  
preproc.run(plugin='MultiProc', plugin_args={'n_procs' : 5})
