import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
from nipype.interfaces.utility import Function
from nipype.interfaces.utility import IdentityInterface as ID
from nipype.pipeline.engine import Workflow, Node, MapNode
from math import pi as pi
import argparse as ap

#Define the directories where data is located
parser = ap.ArgumentParser(description='Preprocess all runs simultaneously for one participant.\nTo be used in conjunction with PBS script, using a -t array for all participants.')
parser.add_argument('-numruns', type=int,
                    help='The number of functional runs (an integer)')  
parser.add_argument('-func', type=str,
                    help="A file containing all of the functional data to be analyzed, with one file path perline") 
parser.add_argument('-str', type=str,
                    help="The path to the structural T1 image") 
parser.add_argument('-mni', type=str,
                    help="The path to the MNI T1 image") 
parser.add_argument('-mni_brain', type=str,
                    help="The path to the brain extracted MNI T1 image")
parser.add_argument('-mni_mask', type=str,
                    help="The path to the mask used to brain extract the MNI T1 image")  
parser.add_argument('-fieldmap_type', action='store_true',
                    help='Takes one of three inputs "gre","se","none":\nthe "gre" option requires a magnitude and phase encoding image \nthe "se" option uses topup and and requires reverse phase encoded images')
parser.add_argument('-hp_cutoff', type=float,
                    help='Highpass filter cuttoff in seconds')
parser.add_argument('-tr', type=float,
                    help='Repetition time in seconds')
parser.add_argument('-fwhm', type=float,
                    help='Smoothing extent in millimeters for spatial smoothing')
parser.add_argument('-outdir', type=str,
                    help='The output directory')
parser.add_argument('-mag', type=str,
                    help="The magnitude image for a gre fieldmap")  
parser.add_argument('-phase',type=str,
                    help='The phase encoding image for a gre fieldmap')
parser.add_argument('-se1', type=str,
                    help='One of the se images for the se fieldmap option')
parser.add_argument('-se2', type=str,
                    help='The se image with the opposite phase encoding direction as se1 for the se fieldmap option')
parser.add_argument('-pe_dir', type=str,
                    help='The phase encoding direction (-x,+x,-y,+y,-z,+z)')
parser.add_argument('-echospacing', type=float,
                    help='The effective echospacing in seconds (a float). See for a discussion https://lcni.uoregon.edu/kb-articles/kb-0003')
parser.add_argument('-datain', type=str,
                    help='File for use with topup containing information on phase encoding direction and the total readout time of se1 on line 1 and se2 on line 2.\n See for a discussion https://lcni.uoregon.edu/kb-articles/kb-0003')                                             
args = parser.parse_args()

if not args.numruns or not args.func or not args.str or not args.mni or not args.mni_brain or not args.mni_mask or not args.fieldmap_type or not args.hp_cutoff or not args.tr or not args.fwhm or not args.outdir:
    print "-numruns, -func, -str, -mni, -mni_brain, -mni_mask, -fieldmap_type, hp_cutoff, -tr, -fwhm, and -outdir must all be specified. They are not optional"
    pass 

numruns = args.numruns
func = [line.rstrip('\n') for line in open(args.func,'r').readlines()]
struct = args.struct
mni = args.mni
mni_brain = args.mni_brain
mni_mask = args.mni_mask
fieldmap_type = args.fieldmap_type
hp_cutoff = args.hp_cutoff
tr = args.tr
fwhm = args.fwhm
outdir = args.outdir

if fieldmap_type == 'gre':
    if not args.mag:
        print "You have indicated that you are using a GRE fieldmap, but have not specified the appropriate files. Please specify the magnitude image."
    if not args.phase:
        print "You have indicated that you are using a GRE fieldmap, but have not specified the appropriate files. Please specify the phase image."
    mag = args.mag
    phase = args.phase
elif fieldmap_type == 'se':
    if not args.se1:
        print "You have indicated that you are using a SE fieldmap, but have not specified the appropriate information. Please specify the se1 image."
    if not args.se2:
        print "You have indicated that you are using a SE fieldmap, but have not specified the appropriate information. Please specify the se2 image."
    if not args.pe_dir:
        print "You have indicated that you are using a SE fieldmap, but have not specified the appropriate information. Please specify the phase encoding direction."
    if not args.echospacing:
        print "You have indicated that you are using a SE fieldmap, but have not specified the appropriate information. Please specify the echospacing."
    if not args.datain:
        print "You have indicated that you are using a SE fieldmap, but have not specified the appropriate information. Please specify the datain file."
    se1 = args.se1
    se2 = args.se2
    pe_dir = args.pe_dir
    echospacing = args.echospacing
    datain = args.datain
    
elif fieldmap_type != 'none':
    print "You have not correctly specified the fieldmap_type argument. Please enter 'gre', 'se', or 'none'"

##-----------------------------------------------------------------##
##This section defines functions that will be used during prerpocessing but
## will not act as actual "nodes" in the preprocessing pipeline        
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
    
def choose_wm(fs):
    return [fs[0][2]]

def choose_csf(fs):
    return [fs[0][0]]
    
def choose_wm_numruns(fs,numruns):
    return [fs[0][2] for i in range(numruns)]

def choose_csf_numruns(fs,numruns):
    return [fs[0][0] for i in range(numruns)]
    
def repfiles(fs,numruns):
    new = []
    for f in fs:
        for i in range(numruns):
            new.append(f)
    return new
##----------------------------------------------------------------##
#This section defines the workflow

#Define Workflow
preproc = Workflow(name = "fMRI_Preprocessing")#,base_dir='/media/bschloss/Extra Drive 1/Preprocessing/')

#Use DataGrabber function to get all of the data
data = Node(ID(fields=['func','struct','mni','mni_brain','highpass_sigma','fwhm','mni_mask','mag','phase','se1','se2','pe_dir','echospacing','datain'],
               mandatory_inputs=False),name='Data')
data.inputs.func = [func]
data.inputs.struct = [struct]
data.inputs.mni = mni
data.inputs.mni_brain = mni_brain
data.inputs.mni_mask = mni_mask
data.inputs.highpass_sigma = float((hp_cutoff/tr)/2.355)
data.inputs.fwhm = fwhm
data.inputs.mag = [mag]
data.inputs.phase = [phase]
data.inputs.se1 = [se1]
data.inputs.se2 = [se2]
data.inputs.pe_dir = pe_dir
data.inputs.echospacing = echospacing
data.inputs.datain = datain

#Use the DataSink function to store all outputs
datasink = Node(nio.DataSink(), name= 'Output')
datasink.inputs.base_directory = outdir

#Convert input data to float representation
img2float = MapNode(interface=fsl.ImageMaths(out_data_type='float',
                                                 op_string = '',
                                                 suffix='_dtype'),
                           iterfield=['in_file'],
                           name='img2float')
                           
preproc.connect(data, 'func', img2float, 'in_file')
if fieldmap_type == 'gre':
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
    
elif fieldmap_type == 'se':
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

if fieldmap_type == 'se':
    #Coregister the mean functional data image to the respective structural image
    register_f2s = MapNode(fsl.EpiReg(),
                           name='Register_F2S',
                           iterfield=['epi','t1_brain','t1_head','fmap','fmapmag','fmapmagbrain','wmseg'])
    preproc.connect(data,'pe_dir',register_f2s,'pedir')
    preproc.connect(data,'echospacing',register_f2s,'echospacing')
    preproc.connect(functional_brain_mask,'out_file',register_f2s,'epi')
    preproc.connect(bet,('out_file',repfiles),register_f2s,'t1_brain')
    preproc.connect(data,('struct',repfiles),register_f2s,'t1_head')
    preproc.connect(fmap2rads,('out_file',repfiles),register_f2s,'fmap')
    preproc.connect(se_mag_mean,('out_file',repfiles),register_f2s,'fmapmag')
    preproc.connect(se_mag_bet,('out_file',repfiles),register_f2s,'fmapmagbrain')
    preproc.connect(fast,('tissue_class_files',choose_wm_numruns),register_f2s,'wmseg')
    preproc.connect(register_f2s,'epi2str_mat',datasink,'F2S_Affine')
    preproc.connect(register_f2s,'out_file',datasink,'Coregistered_F2S')
    preproc.connect(register_f2s,'wmedge',datasink,'WM_Edges')
    preproc.connect(register_f2s,'shiftmap',datasink,'Shift_Map')

elif fieldmap_type == 'gre':
    #Coregister the mean functional data image to the respective structural image
    register_f2s = MapNode(fsl.EpiReg(),
                           name='Register_F2S',
                           iterfield=['epi','t1_brain','t1_head','fmap','fmapmag','fmapmagbrain','wmseg'])
    preproc.connect(functional_brain_mask,'out_file',register_f2s,'epi')
    preproc.connect(bet,('out_file',repfiles),register_f2s,'t1_brain')
    preproc.connect(data,('struct',repfiles),register_f2s,'t1_head')
    preproc.connect(prepare_fm,('out_fieldmap',repfiles),register_f2s,'fmap')
    preproc.connect(gre_mag_mean,('out_file',repfiles),register_f2s,'fmapmag')
    preproc.connect(gre_mag_bet,('out_file',repfiles),register_f2s,'fmapmagbrain')
    preproc.connect(fast,('tissue_class_files',choose_wm_numruns),register_f2s,'wmseg')
    preproc.connect(register_f2s,'epi2str_mat',datasink,'F2S_Affine')
    preproc.connect(register_f2s,'out_file',datasink,'Coregistered_F2S')
    preproc.connect(register_f2s,'wmedge',datasink,'WM_Edges')
    preproc.connect(register_f2s,'shiftmap',datasink,'Shift_Map')

else:
    #Coregister the mean functional data image to the respective structural image
    register_f2s = MapNode(fsl.EpiReg(),
                           name='Register_F2S',
                           iterfield=['epi','t1_brain','t1_head','wmseg'])
    preproc.connect(functional_brain_mask,'out_file',register_f2s,'epi')
    preproc.connect(bet,('out_file',repfiles),register_f2s,'t1_brain')
    preproc.connect(data,('struct',repfiles),register_f2s,'t1_head')
    preproc.connect(fast,('tissue_class_files',choose_wm_numruns),register_f2s,'wmseg')
    preproc.connect(register_f2s,'epi2str_mat',datasink,'F2S_Affine')
    preproc.connect(register_f2s,'out_file',datasink,'Coregistered_F2S')
    preproc.connect(register_f2s,'wmedge',datasink,'WM_Edges')
    
    
#Get average time series of whitematter masked file
avg_wmts = MapNode(fsl.ImageMeants(),name='WM_Time_Series_Averager',
                iterfield=['in_file','mask'])
preproc.connect(register_f2s,'out_file',avg_wmts,'in_file')
preproc.connect(fast,('tissue_class_files',choose_wm_numruns),avg_wmts,'mask')
preproc.connect(avg_wmts,'out_file',datasink,'WM_TS')

#Get average time series of cerebral spinal fluid file
avg_csfts = MapNode(fsl.ImageMeants(),name='CSF_Time_Series_Averager',
                iterfield=['in_file','mask'])
preproc.connect(register_f2s,'out_file',avg_csfts,'in_file')
preproc.connect(fast,('tissue_class_files',choose_csf_numruns),avg_csfts,'mask')
preproc.connect(avg_csfts,'out_file',datasink,'CSF_TS')

eig_wmts = MapNode(fsl.ImageMeants(eig=True,order=3),
		   name='WM_Time_Series_Eigenvariates',
		   iterfield=['in_file','mask'])
preproc.connect(register_f2s,'out_file',eig_wmts,'in_file')
preproc.connect(fast,('tissue_class_files',choose_wm_numruns),eig_wmts,'mask')
preproc.connect(eig_wmts,'out_file',datasink,'WM_Eig')

eig_csfts = MapNode(fsl.ImageMeants(eig=True,order=3),
                   name='CSF_Time_Series_Eigenvariates',
                   iterfield=['in_file','mask'])
preproc.connect(register_f2s,'out_file',eig_csfts,'in_file')
preproc.connect(fast,('tissue_class_files',choose_csf_numruns),eig_csfts,'mask')
preproc.connect(eig_csfts,'out_file',datasink,'CSF_Eig')

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
if fieldmap_type == 'gre' or fieldmap_type == 'se':
    convert_warp = MapNode(fsl.ConvertWarp(relwarp = True, shift_direction='y-'),
                           name='Concat_Shift_Affine_and_Warp',
                           iterfield = ['shift_in_file','premat','warp1'])
    preproc.connect(register_f2s,'shiftmap',convert_warp,'shift_in_file')
    preproc.connect(register_f2s,'epi2str_mat',convert_warp,'premat')
    preproc.connect(register_s2MNI_NL,('field_file',repfiles),convert_warp,'warp1')
    preproc.connect(data,'mni',convert_warp,'reference')
    preproc.connect(convert_warp,'out_file',datasink,'Warp_File_F2MNI') 
    
else:
    convert_warp = MapNode(fsl.ConvertWarp(relwarp = True, shift_direction='y-'),
                           name='Concat_Shift_Affine_and_Warp',
                           iterfield = ['premat','warp1'])
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

#Highpass filter the data (which removes the mean)
highpass = MapNode(fsl.TemporalFilter(),
                   name = 'HighPass_Filter',
                   iterfield = ['in_file']) 
preproc.connect(data,'highpass_sigma',highpass,'highpass_sigma')
preproc.connect(apply_warp,'out_file',highpass,'in_file')

#Add the mean back in
add_mean_back_in = MapNode(fsl.BinaryMaths(operation='add'),
                   name = 'Add_Mean_Back_In',
                   iterfield = ['in_file','operand_file'])
preproc.connect(highpass,'out_file',add_mean_back_in,'in_file')
preproc.connect(meants,'out_file',add_mean_back_in,'operand_file')
preproc.connect(add_mean_back_in,'out_file',datasink,'Highpass')

#Smooth the data                 
smoother = MapNode(interface=fsl.IsotropicSmooth(),
                      name='Smoother',
                      iterfield=['in_file'])
preproc.connect(add_mean_back_in, 'out_file', smoother, 'in_file')
preproc.connect(data,'fwhm',smoother,'fwhm')
preproc.connect(smoother, 'out_file', datasink, 'Smoothed')

preproc.write_graph(dotfilename='fMRI_Preprocessing_Graph.dot',format='svg',graph2use='exec')
preproc.run(plugin='MultiProc', plugin_args={'n_procs' : numruns})