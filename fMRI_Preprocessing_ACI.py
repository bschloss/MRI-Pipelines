import os
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
from nipype.interfaces.utility import Function
from nipype.interfaces.utility import IdentityInterface as ID
from nipype.pipeline.engine import Workflow, Node, MapNode
from math import pi as pi
import argparse as ap

#Define the directories where data is located
parser = ap.ArgumentParser(description='Preprocess DTI data and put in new folder')
parser.add_argument('pardir', metavar="stanford", type=str,
                    help="Path to participant's directory")                                                  
args = parser.parse_args()
par_dir= '/storage/group/pul8_collab/read/'
if len(args.pardir) == 1:
    par_dir += '00' + args.pardir + '/'
elif len(args.pardir) == 2:
    par_dir += '0' + args.pardir + '/'
else:
    par_dir += args.pardir + '/'
fieldmap_dir = par_dir + "Fieldmap/"
func_data_dir = par_dir + "Func/"
str_data_dir = par_dir + "Struct/"

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
data.inputs.func = [f + '/' + os.listdir(f)[0] for f in [func_data_dir + 'Run' + str(i) for i in range(1,6)]]
data.inputs.struct = [str_data_dir + f for f in os.listdir(str_data_dir) if 'co' in f and 'warp' not in f]
data.inputs.mni = '/storage/group/pul8_collab/read/MNI152_T1_3mm3mm4mm.nii.gz'
data.inputs.mni_brain = '/storage/group/pul8_collab/read/MNI152_T1_3mm3mm4mm_brain.nii.gz'
data.inputs.mni_mask = '/storage/group/pul8_collab/read/MNI152_T1_3mm3mm4mm_brain_mask.nii.gz'
data.inputs.mag = [fieldmap_dir + f for f in os.listdir(fieldmap_dir) if '1001.nii.gz' in f]
data.inputs.phase = [fieldmap_dir + f for f in os.listdir(fieldmap_dir) if '2001.nii.gz' in f]
data.inputs.seAP = [fieldmap_dir + f for f in os.listdir(fieldmap_dir) if 'AP' in f and 'Grappa' not in f]
data.inputs.sePA = [fieldmap_dir + f for f in os.listdir(fieldmap_dir) if 'PA' in f and 'Grappa' not in f]
data.inputs.datain = '/storage/group/pul8_collab/read/datain_fMRI.txt'
#Use the DataSink function to store all outputs
datasink = Node(nio.DataSink(), name= 'Output')
datasink.inputs.base_directory = par_dir + 'fMRI_Preprocessing/'

#Convert input data to float representation
img2float = MapNode(interface=fsl.ImageMaths(out_data_type='float',
                                                 op_string = '',
                                                 suffix='_dtype'),
                           iterfield=['in_file'],
                           name='img2float')
                           
preproc.connect(data, 'func', img2float, 'in_file')
'''
gre_mag_mean = MapNode(interface=fsl.MeanImage(),
                 name='GRE_Mean_Mag',
                 iterfield = ['in_file'])
preproc.connect(data,'mag',gre_mag_mean,'in_file')
preproc.connect(gre_mag_mean,'out_file',datasink,'GRE_Magnitude_File')

gre_mag_bet = MapNode(fsl.BET(mask=True),
                     name = 'Brain_Extract_GRE_Mag',
                     iterfield = ['in_file'])
preproc.connect(gre_mag_mean,'out_file',gre_mag_bet,'in_file')
preproc.connect(gre_mag_bet,'out_file',datasink,'Brain_Extracted_GRE_Mag_No_Erosion')

gre_erode_mask = MapNode(fsl.ErodeImage(),
                     name = 'Erode_GRE_Mag_Map',
                     iterfield = ['in_file'])
preproc.connect(gre_mag_bet,'mask_file',gre_erode_mask,'in_file')
preproc.connect(gre_erode_mask,'out_file',datasink,'GRE_Mag_Eroded_Mask')

gre_apply_ero = MapNode(fsl.ApplyMask(),
                    name = "GRE_Apply_Eroded_Brain_Mask",
                    iterfield = ['in_file','mask_file'])
preproc.connect(gre_erode_mask,'out_file',gre_apply_ero,'mask_file')
preproc.connect(gre_mag_mean,'out_file',gre_apply_ero,'in_file')
preproc.connect(gre_apply_ero,'out_file',datasink,'Brain_Extracted_GRE_Mag_Eroded_Mask')

prepare_fm = MapNode(interface=fsl.epi.PrepareFieldmap(),
                     name = 'Prepare_Field_Map',
                     iterfield = ['in_magnitude','in_phase'])
preproc.connect(gre_apply_ero,'out_file',prepare_fm,'in_magnitude')
preproc.connect(data,'phase',prepare_fm,'in_phase')
preproc.connect(prepare_fm,'out_fieldmap',datasink,'GRE_Field_Map')

gre_sig_loss = MapNode(interface=fsl.epi.SigLoss(),
                       name = 'Estimate_Signal_Loss',
                       iterfield = ['in_file'])
preproc.connect(prepare_fm,'out_fieldmap',gre_sig_loss,'in_file')
preproc.connect(gre_sig_loss,'out_file',datasink,'Estimated_Signal_Loss')
'''
AP0 = MapNode(fsl.ExtractROI(t_min=0,t_size=1),
                  name = 'AP0',
                  iterfield = ['in_file'])
preproc.connect(data,'seAP',AP0,'in_file')
preproc.connect(AP0,'roi_file',datasink,'AP0')

PA0 = MapNode(fsl.ExtractROI(t_min=0,t_size=1),
                  name = 'PA0',
                  iterfield = ['in_file'])
preproc.connect(data,'sePA',PA0,'in_file')
preproc.connect(PA0,'roi_file',datasink,'PA0')

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
'''
se_erode_mask = MapNode(fsl.ErodeImage(),
                     name = 'Erode_SE_Mag_Map',
                     iterfield = ['in_file'])
preproc.connect(se_mag_bet,'mask_file',se_erode_mask,'in_file')
preproc.connect(se_erode_mask,'out_file',datasink,'SE_Mag_Eroded_Mask')

se_apply_ero = MapNode(fsl.ApplyMask(),
                    name = "SE_Apply_Eroded_Brain_Mask",
                    iterfield = ['in_file','mask_file'])
preproc.connect(se_erode_mask,'out_file',se_apply_ero,'mask_file')
preproc.connect(se_mag_mean,'out_file',se_apply_ero,'in_file')
preproc.connect(se_apply_ero,'out_file',datasink,'Brain_Extracted_SE_Mag_Eroded_Mask')
'''
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
preproc.connect(fast,('tissue_class_files',choose_wm),datasink,'CSF_Seg')

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
register_f2s = MapNode(fsl.EpiReg(pedir='-y',echospacing = .00058),
                       name='Register_F2S',
                       iterfield=['epi','t1_brain','t1_head','fmap','fmapmag','fmapmagbrain','wmseg'])
preproc.connect(functional_brain_mask,'out_file',register_f2s,'epi')
preproc.connect(bet,('out_file',repfiles),register_f2s,'t1_brain')
preproc.connect(data,('struct',repfiles),register_f2s,'t1_head')
preproc.connect(fmap2rads,('out_file',repfiles),register_f2s,'fmap')
preproc.connect(se_mag_mean,('out_file',repfiles),register_f2s,'fmapmag')
preproc.connect(se_mag_bet,('out_file',repfiles),register_f2s,'fmapmagbrain')
preproc.connect(fast,('tissue_class_files',choose_wm5),register_f2s,'wmseg')
preproc.connect(register_f2s,'epi2str_mat',datasink,'F2S_Affine')
preproc.connect(register_f2s,'out_file',datasink,'Coregistered_F2S')
preproc.connect(register_f2s,'wmedge',datasink,'WM_Edges')
preproc.connect(register_f2s,'shiftmap',datasink,'Shift_Map')

#Get average time series of whitematter masked file
avg_wmts = MapNode(fsl.ImageMeants(),name='WM_Time_Series_Averager',
                iterfield=['in_file','mask'])
preproc.connect(register_f2s,'out_file',avg_wmts,'in_file')
preproc.connect(fast,('tissue_class_files',choose_wm5),avg_wmts,'mask')
preproc.connect(avg_wmts,'out_file',datasink,'WM_TS')

#Get average time series of cerebral spinal fluid file
avg_csfts = MapNode(fsl.ImageMeants(),name='CSF_Time_Series_Averager',
                iterfield=['in_file','mask'])
preproc.connect(register_f2s,'out_file',avg_csfts,'in_file')
preproc.connect(fast,('tissue_class_files',choose_csf5),avg_csfts,'mask')
preproc.connect(avg_csfts,'out_file',datasink,'CSF_TS')

eig_wmts = MapNode(fsl.ImageMeants(eig=True,order=3),
		   name='WM_Time_Series_Eigenvariates',
		   iterfield=['in_file','mask'])
preproc.connect(register_f2s,'out_file',eig_wmts,'in_file')
preproc.connect(fast,('tissue_class_files',choose_wm5),eig_wmts,'mask')
preproc.connect(eig_wmts,'out_file',datasink,'WM_Eig')

eig_csfts = MapNode(fsl.ImageMeants(eig=True,order=3),
                   name='CSF_Time_Series_Eigenvariates',
                   iterfield=['in_file','mask'])
preproc.connect(register_f2s,'out_file',eig_csfts,'in_file')
preproc.connect(fast,('tissue_class_files',choose_csf5),eig_csfts,'mask')
preproc.connect(eig_csfts,'out_file',datasink,'CSF_Eig')

combine_covariates = MapNode(Function(input_names = ['mp','csf','wm'],
                        output_names = ['covariates'],
                        function = combine_covariates),
                name = 'Combine_Covariates',
                iterfield=['mp','csf','wm'])
preproc.connect(motion_correct,'par_file',combine_covariates,'mp')
preproc.connect(eig_csfts,'out_file',combine_covariates,'csf')
preproc.connect(eig_wmts,'out_file',combine_covariates,'wm')
preproc.connect(combine_covariates,'covariates',datasink,'Motion_CSF_WM_Covariates')

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
                                      skip_inmask = True),
                                      #apply_refmask = [0,0,0,1,1,1],
                                      #regularization_lambda=[400,200,150,75,60,45],
                                      #subsampling_scheme = [8,4,2,2,1,1],
                                      #max_nonlin_iter = [5,5,5,10,5,10],
                                      #in_fwhm = [8,6,4,2,2,2],
                                      #ref_fwhm = [6,4,2,0,0,0],
                                      #apply_intensity_mapping = [1,1,1,1,1,0]),
                            name='Registration_s2MNI_NL',
                            iterfield=['in_file','affine_file'])
preproc.connect(data,'struct',register_s2MNI_NL,'in_file')
preproc.connect(register_s2MNI_L,'out_matrix_file',register_s2MNI_NL,'affine_file')
preproc.connect(data,'mni',register_s2MNI_NL,'ref_file')
preproc.connect(data,'mni_mask',register_s2MNI_NL,'refmask_file')
preproc.connect(register_s2MNI_NL,'field_file',datasink,'Warp_Field_File')
preproc.connect(register_s2MNI_NL,'fieldcoeff_file',datasink,'Warp_Field_Coeff')
preproc.connect(register_s2MNI_NL,'warped_file',datasink,'Registered_s2MNI_NL_Warped')

#Concatenate the shiftmap, functional to structural affine, and stuctural to mni warp
#into a single transform.
convert_warp = MapNode(fsl.ConvertWarp(relwarp = True, shift_direction='y-'),
                       name='Concat_Shift_Affine_and_Warp',
                       iterfield = ['shift_in_file','premat','warp1'])
preproc.connect(register_f2s,'shiftmap',convert_warp,'shift_in_file')
preproc.connect(register_f2s,'epi2str_mat',convert_warp,'premat')
preproc.connect(register_s2MNI_NL,('field_file',repfiles),convert_warp,'warp1')
preproc.connect(data,'mni',convert_warp,'reference')
preproc.connect(convert_warp,'out_file',datasink,'Warp_File_F2MNI') 

#Apply warp to functional data that has been registered to structural image
#converting it to MNI space using Nonlinear registartion
apply_warp = MapNode(fsl.ApplyWarp(),name='Warper_F2MNI',
                     iterfield=['in_file','field_file'])
preproc.connect(data,'mni',apply_warp,'ref_file')
preproc.connect(convert_warp,'out_file',apply_warp,'field_file')
preproc.connect(data,'mni_mask',apply_warp,'mask_file')
preproc.connect(motion_correct,'out_file',apply_warp,'in_file')
preproc.connect(apply_warp,'out_file',datasink,'Registered_F2MNI_NL_Warped')

#Get mean-image in time direction           
meants = MapNode(fsl.MeanImage(),
                name= 'Mean_Func',
                iterfield=['in_file'])
preproc.connect(apply_warp,'out_file',meants,'in_file')

#Regress out task regressors and WM and CS regressors
filter_regressor = MapNode(fsl.FilterRegressor(filter_all=True),
                           name="Filter_Regressor",
                           iterfield=['design_file','in_file'])
preproc.connect(apply_warp,'out_file',filter_regressor,'in_file')
preproc.connect(combine_covariates,'covariates',filter_regressor,'design_file')
preproc.connect(filter_regressor,'out_file',datasink,'Filtered')

#Highpass filter the data (which removes the mean)
highpass = MapNode(fsl.TemporalFilter(highpass_sigma = float((128.0/.400)/2.355)),
                   name = 'HighPass_Filter',
                   iterfield = ['in_file']) 
preproc.connect(apply_warp,'out_file',highpass,'in_file')

#Add the mean back in
add_mean_back_in = MapNode(fsl.BinaryMaths(operation='add'),
                   name = 'Add_Mean_Back_In',
                   iterfield = ['in_file','operand_file'])
preproc.connect(highpass,'out_file',add_mean_back_in,'in_file')
preproc.connect(meants,'out_file',add_mean_back_in,'operand_file')
preproc.connect(add_mean_back_in,'out_file',datasink,'Highpass')

#Smooth the data                 
smoother = MapNode(interface=fsl.IsotropicSmooth(fwhm=8),
                      name='Smoother',
                      iterfield=['in_file'])
preproc.connect(add_mean_back_in, 'out_file', smoother, 'in_file')
preproc.connect(smoother, 'out_file', datasink, 'Smoothed')

#Highpass filter the data (which removes the mean)
highpass_filtered = MapNode(fsl.TemporalFilter(highpass_sigma = float((128.0/.400)/2.355)),
                   name = 'HighPass_Filter_Filtered',
                   iterfield = ['in_file']) 
preproc.connect(filter_regressor,'out_file',highpass_filtered,'in_file')

#Add the mean back in
add_mean_back_in_filtered = MapNode(fsl.BinaryMaths(operation='add'),
                   name = 'Add_Mean_Back_In_Filtered',
                   iterfield = ['in_file','operand_file'])
preproc.connect(highpass_filtered,'out_file',add_mean_back_in_filtered,'in_file')
preproc.connect(meants,'out_file',add_mean_back_in_filtered,'operand_file')
preproc.connect(add_mean_back_in_filtered,'out_file',datasink,'Highpass_Filtered')

#Smooth the data                 
smoother_filtered = MapNode(interface=fsl.IsotropicSmooth(fwhm=8),
                      name='Smoother_Filtered',
                      iterfield=['in_file'])
preproc.connect(add_mean_back_in_filtered, 'out_file', smoother_filtered, 'in_file')
preproc.connect(smoother_filtered, 'out_file', datasink, 'Smoothed_Filtered')

#preproc.write_graph(dotfilename='fMRI_Preprocessing_Graph.dot',format='svg')
#preproc.write_graph(dotfilename='fMRI_Preprocessing_Graph.dot',format='svg',graph2use='exec')
preproc.run(plugin='MultiProc', plugin_args={'n_procs' : 5})

