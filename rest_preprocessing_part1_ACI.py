# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:11:21 2015

@author: bschloss
"""
import os
import nipype.interfaces.io as nio
import nipype.interfaces.afni as afni
import nipype.interfaces.fsl as fsl
from nipype.interfaces.utility import IdentityInterface as ID
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import Function
import argparse as ap
import exceptions

##-----------------------------------------------------------------##
##This section defines functions that will be used during prerpocessing but
## will not act as actual "nodes" in the preprocessing pipeline
def get_crash(crashdir):
    import os
    import gzip as gz
    import pickle as pkl
    crash = [crashdir + f for f in os.listdir(crashdir) if 'crash' in f]
    return [pkl.load(gz.open(f)) for f in crash]

def physio_file(resp,pulse,triggers):
	#puts the phyio data in a format FSL reads
	#the sampling rates for the respiratory pulse is diff which is why this is so giant.
    #Timing in this files is in terms of tics.
    #The value of each tic seems to be 2.5ms
    #There is a 55ms discrepency between the difference between
    #The onset of slice 1 and offset of slice 32 (199.945ms) and
    #The time of the total volume acquistiion (200.000ms). It is ambiguous
    #Where the actual start and end of each TR begins.
    import os

    out_base = os.getcwd() #gets the output folder
    time_step = 8 #One line every 8 tics [realigning; a tic is 2.5 ms--we down-sample the faster one]
    t = open(triggers,'r').readlines()
    p = open(pulse,'r').readlines()
    r = open(resp,'r').readlines()
    physio = '' #stuff the tigger, pulse and resp into this string

    t_line  = 0 #gives trigger file line number; keep track of them
    while(t_line<len(t)):
        if t[t_line].split(): #split line by tabs or spaces
            if t[t_line].split()[0]=='0':
                break
        t_line += 1
    tzero = int(t[t_line].split()[-2]) - int(34/2.5) #Account for 34 ms before the start of the first slice
    #there's a readout time the scanner goes through, and this is what the trigger is for
    #scan itself starts 34 ms before the first readout (the time point we need)
    #we're manually setting tzero to be the tic space (divided by 2.5 ms per tic) when the scan starts

    p_line = 0 #line the pulse file with tzero
    while(p_line < len(p)):
        if p[p_line].split():
            if p[p_line].split()[1]=='PULS':
                break
        p_line += 1

    r_line = 0 #same with resp
    while(r_line < len(r)):
        if r[r_line].split():
            if r[r_line].split()[1]=='RESP':
                break
        r_line += 1

    trigger = tzero
    while(trigger < (tzero + int(150*2*1000/2.5))): #conversion to tic space: vol num * TR * ms per tic
        while(r_line < len(r) - 1 and
              abs(int(r[r_line+1].split()[0]) - trigger) < abs(int(r[r_line].split()[0]) - trigger)): #we're finding the resp sampling points that are the closest in time to our target tics
            r_line += 1 #if the next time point will get us closer to the next 8 tic time-step then go to the next one
        while(p_line < len(p) - 1 and  #repeat the same step for pulse data
              abs(int(p[p_line+1].split()[0]) - trigger) < abs(int(p[p_line].split()[0]) - trigger)):
            p_line += 1

        rval = int(r[r_line].split()[2]) #create new line in physio file
        pval = int(p[p_line].split()[2]) #pull out continous signal value from this line for resp and pulse
        if (trigger-tzero)%800==0: #800*2.5=2000 ms --> TR; every new slice we have the start of the new readout, in that case tval is 1
            tval = 1
        else:
            tval = 0
        physio += '\t'.join([str(tval),str(pval),str(rval)]) + '\n' #make new line denoting 1. is it the start of a new readout, 2. value of pulse 3. value of resp
        trigger += time_step
    physio = physio.rstrip('\n')
    physiofname = out_base + '/physio.txt'
    open(physiofname,'w').write(physio)
    return physiofname

def PNM(physio,func,st): #takes the physio file, the fxnal file, and the slice timing file
    import subprocess as sp #allows to call a terminal from within python
    import os
    out_base = os.getcwd()
    command = ['/gpfs/group/pul8/default/fsl/bin/fslFixText',physio,'/'.join([out_base,'input.txt'])]#fsl fxn that formats the physio file
    fixinput = sp.Popen(command) #send command to terminal
    sp.Popen.wait(fixinput) #wait until done

    command = ['/gpfs/group/pul8/default/fsl/bin/popp',
               '-i','/'.join([out_base,'input.txt']),
               '-o', '/'.join([out_base,'Physio']),
               '-s',
               str(50),
               '--tr=2.000000',
               '--smoothcard=0.1',
               '--smoothresp=0.1',
               '--resp=3',
               '--cardiac=2',
               '--trigger=1',
               '--startingsample=0',
               '--rvt',
               '--heartrate']   #takes our physio and resp data and outputs other physio info (heart rate, cardio... etc.)
    popp = sp.Popen(command)
    sp.Popen.wait(popp)

    card = out_base + '/Physio_card.txt'
    resp = out_base + '/Physio_resp.txt'
    rvt = out_base + '/Physio_rvt.txt'
    hr = out_base + '/Physio_hr.txt'
    time = out_base + '/Physio_time.txt'
    command = ['/gpfs/group/pul8/default/fsl/bin/pnm_evs',#takes the physio data and creates voxel-wise confound regressors
               '-i',func,
               '-c', card,
               '-r', resp,
               '='.join(['--rvt',rvt]),
               '='.join(['--heartrate',hr]),
               '-o', '/'.join([out_base,'Physio_']),
               '--tr=2.00',
               '--oc=4',
               '--or=4',
               '--multc=2',
               '--multr=2',
               ''.join(['--slicetiming=',st])]
    pnm_evs = sp.Popen(command)
    sp.Popen.wait(pnm_evs)

    evlist = sorted([f for f in os.listdir(out_base) if 'nii.gz' in f])
    evlist = [out_base + '/' + f for f in evlist]
    evlist_txt = ''
    for f in evlist:
        evlist_txt += f + '\n'
    evlist_txt = evlist_txt.rstrip('\n')
    evlist_fname  = out_base + '/Physio_evlist.txt'
    open(evlist_fname,'w').write(evlist_txt)
    return evlist,evlist_fname,card,rvt,hr,time #gather and reorganize data; this fxn will return all of these.

def graph_physio(physio,card,rvt,hr,time): #graphs the data
    import os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_base = os.getcwd()
    time = [float(line.rstrip('\r\n')) for line in open(time,'r').readlines()]
    xtics_major = np.arange(0,300,.20)

    cardio = [float(line.rstrip('\r\n').split('\t')[1]) for line in open(physio,'r').readlines()]
    cardiopeaks = [float(line.rstrip('\r\n')) for line in open(card,'r').readlines()]
    plt.figure(figsize=(150,10))
    plt.plot(time,cardio,'r-',lw=1.0)
    for x in cardiopeaks:
        plt.axvline(x, color='k', linestyle='-')
    plt.ylabel('Cardio')
    plt.xlabel('Time')
    plt.tick_params(axis='x', which='major', labelsize=5)
    plt.xticks(xtics_major)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    plt.title('Check_Cardio_Peaks')
    plt.savefig('/'.join([out_base,'cardio_peaks.png']))
    plt.clf()

    resp = [float(line.rstrip('\r\n').split('\t')[2]) for line in open(physio,'r').readlines()]
    resppeaks = [float(line.rstrip('\r\n').split()[0]) for line in open(rvt,'r').readlines()]
    plt.figure(figsize=(150,10))
    plt.plot(time,resp,'r-',lw=1.0)
    for x in resppeaks:
        plt.axvline(x, color='k', linestyle='-')
    plt.ylabel('Resp')
    plt.xlabel('Time')
    plt.tick_params(axis='x', which='major', labelsize=5)
    plt.xticks(xtics_major)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    plt.title('Check_RVT_Peaks')
    plt.savefig('/'.join([out_base,'rvt_peaks.png']))
    plt.clf()

    cardio = [float(line.rstrip('\r\n').split('\t')[1]) for line in open(physio,'r').readlines()]
    hrpeaks = [float(line.rstrip('\r\n').split()[0]) for line in open(hr,'r').readlines()]
    plt.figure(figsize=(150,10))
    plt.plot(time,cardio,'r-',lw=1.0)
    for x in hrpeaks:
        plt.axvline(x, color='k', linestyle='-')
    plt.ylabel('Cardio')
    plt.xlabel('Time')
    plt.tick_params(axis='x', which='major', labelsize=5)
    plt.xticks(xtics_major)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    plt.title('Check_Heartrate_Peaks')
    plt.savefig('/'.join([out_base,'heartrate_peaks.png']))
    plt.clf()
    return ['/'.join([out_base,'cardio_peaks.png']),'/'.join([out_base,'rvt_peaks.png']),'/'.join([out_base,'heartrate_peaks.png'])]

def list_of_lists(in_file): #specific to the nypype pipeline which helps to create multiple software pipelines
    return [[item] for item in in_file]

def choose_wm(fs):
    return [fs[0][2]] #grabs the white matter file from fast (segmentation)

def choose_csf(fs):
    return [fs[0][0]] #grabs csf


#Define the directories where data is located
parser = ap.ArgumentParser(description='Rest Preprocessing Part 1')
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
#pardir = '/gpfs/group/pul8/default/read/002/'
rest_data_dir= pardir + "Rest/"
str_data_dir = pardir + "Struct/"
physio_dir = pardir + "Physio/"


##----------------------------------------------------------------##
#This section defines the workflow

#Initiate and name Workflow
preproc = Workflow(name = "Rest_Preprocessing_Part_One")#,base_dir='/media/bschloss/Extra Drive 1/Preprocessing/')

#Put all of the data in the input node
data = Node(ID(fields=['rest','struct','pulse','resp','triggers','st','mni','mni_brain','mni_mask'],
               mandatory_inputs=False),name='Data')
data.inputs.rest = [rest_data_dir + f for f in os.listdir(rest_data_dir)]
data.inputs.struct = [str_data_dir + f for f in os.listdir(str_data_dir) if 'co' in f and 'warp' not in f]
data.inputs.pulse = [physio_dir + f for f in os.listdir(physio_dir) if 'PULS' in f]
data.inputs.resp = [physio_dir + f for f in os.listdir(physio_dir) if 'RESP' in f]
data.inputs.triggers = [physio_dir + f for f in os.listdir(physio_dir) if 'Info' in f]
data.inputs.st = '/gpfs/group/pul8/default/read/slicetiming_rest.txt'
data.inputs.mni = '/gpfs/group/pul8/default/read/MNI152_T1_3mm3mm4mm.nii.gz'
data.inputs.mni_brain = '/gpfs/group/pul8/default/read/MNI152_T1_3mm3mm4mm_brain.nii.gz'
data.inputs.mni_mask = '/gpfs/group/pul8/default/read/MNI152_T1_3mm3mm4mm_brain_mask.nii.gz'

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

#Despike data
despike = MapNode(afni.Despike(), name='Despiker',iterfield=['in_file'])
despike.inputs.outputtype = 'NIFTI_GZ'
preproc.connect(img2float,'out_file',despike,'in_file')

#Motion correct the functional data
motion_correct = MapNode(interface=fsl.MCFLIRT(save_mats = True,
                                               save_plots = True,
                                               stats_imgs = True,
                                               mean_vol = True),
                            name='realign',
                            iterfield = ['in_file'])

preproc.connect(despike, 'out_file', motion_correct, 'in_file')
preproc.connect(motion_correct, 'out_file',datasink,'Motion_Corrected')
preproc.connect(motion_correct, 'par_file', datasink,'MotionPars')

#Plot the motion parameters
plot_motion = MapNode(interface=fsl.PlotMotionParams(in_source='fsl',
                                                     plot_type='displacement'),
                            name='plot_motion',
                            iterfield=['in_file'])
preproc.connect(motion_correct, 'par_file', plot_motion, 'in_file')
preproc.connect(plot_motion, 'out_file', datasink, 'Motion_Plots')

#Get mean image of motion corrected data
mean_img = MapNode(fsl.MeanImage(),name='Mean_Image',iterfield=['in_file'])
preproc.connect(motion_correct,'out_file',mean_img,'in_file')

#Skull strip the mean functional image
mean_strip = MapNode(fsl.BET(robust=True,mask=True),name='Mean_Strip', iterfield=['in_file'])
preproc.connect(mean_img,'out_file',mean_strip, 'in_file')
preproc.connect(mean_strip,'out_file', datasink,'Mean_Stripped_Image')

#Apply the mask from mean brain extracted func image
functional_brain_mask = MapNode(fsl.ApplyMask(),
                                name='Functional_Brain_Extracted',
                                iterfield=['in_file','mask_file'])
preproc.connect(mean_strip,'mask_file',functional_brain_mask,'mask_file')
preproc.connect(motion_correct,'out_file',functional_brain_mask,'in_file')
preproc.connect(functional_brain_mask,'out_file', datasink, 'Skull_Stripped_Func')

#Brain Extract structural images
bet = MapNode(fsl.BET(robust=True),name='Brain_Extractor',iterfield=['in_file'])
preproc.connect(data,'struct',bet,'in_file')
preproc.connect(bet,'out_file',datasink,'Brain_Extracted')

#Segment the white matter
fast = MapNode(fsl.FAST(img_type=1,no_pve=True,segments=True),
               name = 'Segmenter',
               iterfield = ['in_files'])

preproc.connect(bet,'out_file',fast,'in_files')
preproc.connect(fast,('tissue_class_files',choose_wm),datasink,'WM_Mask')
preproc.connect(fast,('tissue_class_files',choose_csf),datasink,'CSF_Mask')

#Coregister the mean functional data image to the respective structural image
register_f2s = MapNode(fsl.EpiReg(),
                       name='Register_F2S',
                       iterfield=['epi','t1_head','t1_brain','wmseg'])
preproc.connect(functional_brain_mask,'out_file',register_f2s,'epi')
preproc.connect(bet,'out_file',register_f2s,'t1_brain')
preproc.connect(data,'struct',register_f2s,'t1_head')
preproc.connect(fast,('tissue_class_files',choose_wm),register_f2s,'wmseg')
preproc.connect(register_f2s,'out_file',datasink,'Coregistered_F2S')
preproc.connect(register_f2s,'epi2str_inv',datasink,'Register_F2S_Log')
preproc.connect(register_f2s,'epi2str_mat',datasink,'Affine_F2S')

#Get average time series of whitematter masked file
avg_wmts = MapNode(fsl.ImageMeants(),name='WM_Time_Series_Averager',
                iterfield=['in_file','mask'])
preproc.connect(register_f2s,'out_file',avg_wmts,'in_file')
preproc.connect(fast,('tissue_class_files',choose_wm),avg_wmts,'mask')
preproc.connect(avg_wmts,'out_file',datasink,'WM_TS')

#Get average time series of cerebral spinal fluid file
avg_csfts = MapNode(fsl.ImageMeants(),name='CSF_Time_Series_Averager',
                iterfield=['in_file','mask'])
preproc.connect(register_f2s,'out_file',avg_csfts,'in_file')
preproc.connect(fast,('tissue_class_files',choose_csf),avg_csfts,'mask')
preproc.connect(avg_csfts,'out_file',datasink,'CSF_TS')

physio = MapNode(Function(input_names=['resp','pulse','triggers'],
                                    output_names=['physiofname'],
                                    function=physio_file),
                                    name='Physio_File',
                                    iterfield=['resp','pulse','triggers'])
preproc.connect(data,'resp',physio,'resp')
preproc.connect(data,'pulse',physio,'pulse')
preproc.connect(data,'triggers',physio,'triggers')
preproc.connect(physio,'physiofname',datasink,'Physio_File')

pnm = MapNode(Function(input_names=['physio','func','st'],
                       output_names = ['evlist','evlist_fname','card','rvt','hr','time'],
                       function = PNM),
                       name = 'PNM',
                       iterfield = ['physio','func'])
preproc.connect(data,'st',pnm,'st')
preproc.connect(physio,'physiofname',pnm,'physio')
preproc.connect(motion_correct,'out_file',pnm,'func')
preproc.connect(pnm,'evlist',datasink,'PNM_EVs')

plot_physio = MapNode(Function(input_names=['physio','card','rvt','hr','time'],
                               output_names = ['plots'],
                               function = graph_physio),
                               name = "Physio_Plots",
                               iterfield = ['physio','card','rvt','hr','time'])
preproc.connect(physio,'physiofname',plot_physio,'physio')
preproc.connect(pnm,'card',plot_physio,'card')
preproc.connect(pnm,'rvt',plot_physio,'rvt')
preproc.connect(pnm,'hr',plot_physio,'hr')
preproc.connect(pnm,'time',plot_physio,'time')
preproc.connect(plot_physio,'plots',datasink,'Physio_Graphs')

#Now we register the functional data to MNI space
#Register the structural data to MNI using FLIRT
#this calc the initial linear transform
register_s2MNI_L = MapNode(fsl.FLIRT(dof=6),
                           name='Registration_s2MNI_L',
                           iterfield=['in_file'])
preproc.connect(bet,'out_file',register_s2MNI_L,'in_file')
preproc.connect(data,'mni_brain',register_s2MNI_L,'reference')
preproc.connect(register_s2MNI_L,'out_matrix_file',datasink,'Affine_s2MNI')

#Register the structural data to MNI using FNIRT
#this calc the nonlinear transformation
register_s2MNI_NL = MapNode(fsl.FNIRT(field_file=True,
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
preproc.connect(register_s2MNI_NL,'warped_file',datasink,'Registered_s2MNI_NL_Warped')

#Concatenate

#Concatenate functional to structural affine and stuctural to mni warp
#into a single transform.
convert_warp = MapNode(fsl.ConvertWarp(relwarp = True),
                       name='Concat_Shift_Affine_and_Warp',
                       iterfield = ['premat','warp1'])
preproc.connect(register_f2s,'epi2str_mat',convert_warp,'premat')
preproc.connect(register_s2MNI_NL,'field_file',convert_warp,'warp1')
preproc.connect(data,'mni',convert_warp,'reference')
preproc.connect(convert_warp,'out_file',datasink,'Warp_File_F2MNI')

#Apply warp to functional data that has been registered to structural image
#converting it to MNI space using Nonlinear registartion
#warps
apply_warp = MapNode(fsl.ApplyWarp(),name='Warper_F2MNI',
                     iterfield=['in_file','field_file'])
preproc.connect(data,'mni',apply_warp,'ref_file')
preproc.connect(convert_warp,'out_file',apply_warp,'field_file')
preproc.connect(data,'mni_mask',apply_warp,'mask_file')
preproc.connect(motion_correct,'out_file',apply_warp,'in_file')
preproc.connect(apply_warp,'out_file',datasink,'Registered_F2MNI_NL_Warped')

try:
    preproc.run(plugin='MultiProc', plugin_args={'n_procs' : 3})
except:
    exceptions.OSError

