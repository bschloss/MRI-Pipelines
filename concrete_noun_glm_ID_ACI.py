# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:42:04 2017

@author: bschloss
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 03:39:49 2017

@author: bschloss
"""

import os
import shutil
import string
import exceptions
import numpy.random as random
import nipype.interfaces.io as nio
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface as ID
from nipype.interfaces.utility import Function
import nipype.interfaces.fsl as fsl
import argparse as ap

def nest_list(in_file):
    if len(in_file) == 1:
        out = in_file
    else:
        out = [in_file]
    print in_file
    print out
    return out
    
def unnest_list(in_file):
    if len(in_file[0]) == 1:
	out = in_file
    else:
	out = in_file[0]
    print in_file
    print out
    return out
    
def motionpar_trimmer(mp,ftp):
    import os
    import string
    import exceptions
    import numpy.random as random
    mp_trimmed = open(mp,'r').readlines()[:ftp]
    trimmed = ''
    for line in mp_trimmed:
		trimmed += line
    trimmed = trimmed.rstrip('\n')
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
    file_name = out_base + '/' + mp.split('/')[-1].replace('.par','_trimmed.par')
    with open(file_name,'w') as f:
        f.write(trimmed)
    return os.path.abspath(file_name)

def ID_Feat(parfsf,rest,numevs): #it takes the participant-specific file of how the model should be run and the rest data
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
                  '--con=design.con'])
    p.wait()
    for f in os.listdir(os.getcwd()):
        if f[-3:] != 'fsf':
            sh.move(f,'/'.join([out_base,f]))
    copes = ['stats/cope' + str(i) + '.nii.gz' for i in range(1,numevs+1)] #we want to coregister and warp the copes, the residual and the varcopes
    copes = ['/'.join([out_base,f]) for f in copes]
    dof_file = out_base + '/stats/dof.nii.gz'
    param_estimates = ['stats/pe' + str(i) + '.nii.gz' for i in range(1,numevs+1)]
    param_estimates = ['/'.join([out_base,f]) for f in param_estimates]
    residual4d = out_base + '/stats/res4d.nii.gz' #want to coregister, warp, smooth and bandpass this later
    sigmasquareds = out_base + '/stats/sigmasquareds.nii.gz'
    thresholdac = out_base + '/stats/threshac1.nii.gz'
    tstats = ['stats/tstat' + str(i) + '.nii.gz' for i in range(1,numevs+1)]
    tstats = ['/'.join([out_base,f]) for f in tstats]
    varcopes = ['stats/varcope' + str(i) + '.nii.gz' for i in range(1,numevs+1)]
    varcopes = ['/'.join([out_base,f]) for f in varcopes]
    zstats = ['stats/zstat' + str(i) + '.nii.gz' for i in range(1,numevs+1)]
    zstats = ['/'.join([out_base,f]) for f in zstats]
    return copes,dof_file,param_estimates,residual4d,sigmasquareds,thresholdac,tstats,varcopes,zstats
            
#Define the directories where data is located
parser = ap.ArgumentParser(description='Preprocess DTI data and put in new folder')
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

#Collect and sort event files for model
target = ['axial',
          'battery',
          'boiler',
          'building',
          'canyon',
          'cheese',
          'circle',
          'circuit',
          'coast',
          'combination',
          'current',
          'distance',
          'dust',
          'earth',
          'electric',
          'electron',
          'engineer',
          'gps',
          'gravity',
          'ground',
          'hull',
          'human',
          'instrument',
          'item',
          'lightbulb',
          'live',
          'location',
          'loop',
          'mars',
          'number',
          'ocean',
          'oil',
          'orbit',
          'order',
          'pasta',
          'path',
          'pepperoni',
          'permutation',
          'piece',
          'pizza',
          'plant',
          'propeller',
          'radio',
          'receiver',
          'rice',
          'rover',
          'safety',
          'satellite',
          'sausage',
          'scientist',
          'select',
          'set',
          'ship',
          'shore',
          'signal',
          'source',
          'space',
          'spacecraft',
          'spill',
          'station',
          'storm',
          'sun',
          'temperature',
          'tilt',
          'time',
          'wave',
          'wire',
          'world']
        
target = ['_'.join([word,'target']) for word in target] 
tcon_no_interest_gtbl = ('No_Interest>Baseline','T',['No_Interest'],[1]) 
ev_dir = pardir + 'EV2/'
FSFDIR = pardir + 'FSF/'
try:
    os.mkdir(FSFDIR)
except:
    exceptions.OSError
FSFDIR = FSFDIR + 'IFID/'
try:
    os.mkdir(FSFDIR)
except:
    exceptions.OSError

for word in target:   
    tcon_not_target_fixation_gtbl = ('_'.join([word.replace('_target',''),'Other_Content_or_Target>Baseline']),'T',['_'.join([word.replace('_target',''),'Other_Content_or_Target'])],[1])
    runnum = 0
    for i in range(1,6):
        FSFDIR += 'Run' + str(i) + '/'
        try:
            os.mkdir(FSFDIR)
        except:
            exceptions.OSError
        ev_file = pardir + 'EV2/Run' + str(i) + '/' + word + '1.run00' + str(i) + '.txt'
        func = []
        mp = []
        ev_target = []
        func_trim_point = []
        if os.path.isfile(ev_file):
            #Create a temporary directory
            out_base = ''
            while out_base == '':
                letters = string.ascii_letters
                out_base = ''.join(['/tmp/tmp',''.join([letters[j] for j in random.choice(len(letters),10)])])
                out_base += '/Rename_EVs/'
                try:
                    os.makedirs(out_base)
                except:
                    exceptions.OSError
                    out_base = ''       
            target_contrasts = [tcon_no_interest_gtbl,tcon_not_target_fixation_gtbl] 
            #Only include runs where the participants fixated on that word
            if not (pardir[-3:-1]=='21' and i == 4):
                runnum += 1
                os.makedirs(''.join([out_base,'Run1/']))
                sd = ev_dir + 'Run' + str(i) + '/'
                dd = out_base + 'Run1/'
                #Rename EVs to have the correct number of runs and store in temporary direcotry
                run = []
                for l in ['_'.join([word.replace('_target',''),'Other_Content_or_Target']),'No_Interest']:
                    f_src = sd + l + '.run00' + str(i) + '.txt'
                    f_dest = dd + l + '.run001.txt'
                    shutil.copy(f_src,f_dest)
                    run.append(f_dest)
                fix_num = 1
                while (os.path.isfile(''.join([sd,word,str(fix_num),'.run00',str(i),'.txt']))):
                    f_src = sd + word + str(fix_num) + '.run00' + str(i) + '.txt'
                    f_dest = dd + word + str(fix_num) + '.run001.txt'
                    shutil.copy(f_src,f_dest)
                    run.append(f_dest)
                    tcon_target_word_fixation_gtbl = (''.join([word,str(fix_num),'>Baseline']),'T',[''.join([word,str(fix_num)])],[1])
                    target_contrasts.append(tcon_target_word_fixation_gtbl)
                    fix_num += 1
                ev_target.append(run)
                trim_vol_file = sd + 'text_end_vol_num.run00' + str(i) + '.txt'
                func_trim_point.append(int(open(trim_vol_file,'r').read()))
                f = ''.join([pardir,'fMRI_Preprocessing/Trimmed_Func_Highpass/ _Question_Trimmer_Highpass',str(i-1),'/'])
                f += os.listdir(f)[0]
                func.append(f)
                m = ''.join([pardir,'fMRI_Preprocessing/Trimmed_MotionPars/_Motion_Parameter_Trimmer',str(i-1),'/'])
                m += os.listdir(m)[0]
                mp.append(m)
        
        fsf = ''.join(open('/gpfs/group/pul8/default/read/Scripts/ID.fsf', 'r').readlines()[:277])
        parfsf = fsf.replace('PARNUMBER', str(pardir[-4:-1]))
        parfsf = parfsf.replace('EPI',func[0])
        parfsf = parfsf.replace('MOTIONPARAMETERS',mp[0])
        open("".join([FSFDIR,'design.fsf']), 'w').write(parfsf)
        parfsf = "".join([FSFDIR,'design.fsf'])
        
        id_temp = ''
        id_temp += '# Basic waveform shape (EV 1)\n'
        id_temp += '# 0 : Square\n'
        id_temp += '# 1 : Sinusoid\n'
        id_temp += '# 2 : Custom (1 entry per volume)\n'
        id_temp += '# 3 : Custom (3 column format)\n'
        id_temp += '# 4 : Interaction\n'
        id_temp += '# 10 : Empty (all zeros)\n'
        id_temp += 'set fmri(shape1) 3\n'
        id_temp += '\n'
        id_temp += '# Convolution (EV 1)\n'
        id_temp += '# 0 : None\n'
        id_temp += '# 1 : Gaussian\n'
        id_temp += '# 2 : Gamma\n'
        id_temp += '# 3 : Double-Gamma HRF\n'
        id_temp += '# 4 : Gamma basis functions\n'
        id_temp += '# 5 : Sine basis functions\n'
        id_temp += '# 6 : FIR basis functions\n'
        id_temp += 'set fmri(convolve1) 2\n'
        id_temp += '\n'
        id_temp += '# Convolve phase (EV 1)\n'
        id_temp += 'set fmri(convolve_phase1) 0\n'
        id_temp += '\n'
        id_temp += '# Apply temporal filtering (EV 1)\n'
        id_temp += 'set fmri(tempfilt_yn1) 1\n'
        id_temp += '\n'
        id_temp += '# Add temporal derivative (EV 1)\n'
        id_temp += 'set fmri(deriv_yn1) n'
        
        dg_temp = ''
        dg_temp += '# Basic waveform shape (EV 2)\n'
        dg_temp += '# 0 : Square\n'
        dg_temp += '# 1 : Sinusoid\n'
        dg_temp += '# 2 : Custom (1 entry per volume)\n'
        dg_temp += '# 3 : Custom (3 column format)\n'
        dg_temp += '# 4 : Interaction\n'
        dg_temp += '# 10 : Empty (all zeros)\n'
        dg_temp += 'set fmri(shape2) 3\n'
        dg_temp += '\n'
        dg_temp += '# Convolution (EV 2)\n'
        dg_temp += '# 0 : None\n'
        dg_temp += '# 1 : Gaussian\n'
        dg_temp += '# 2 : Gamma\n'
        dg_temp += '# 3 : Double-Gamma HRF\n'
        dg_temp += '# 4 : Gamma basis functions\n'
        dg_temp += '# 5 : Sine basis functions\n'
        dg_temp += '# 6 : FIR basis functions\n'
        dg_temp += 'set fmri(convolve2) 3\n'
        dg_temp += '\n'
        dg_temp += '# Convolve phase (EV 2)\n'
        dg_temp += 'set fmri(convolve_phase2) 0\n'
        dg_temp += '\n'  
        dg_temp += '# Apply temporal filtering (EV 2)\n'
        dg_temp += 'set fmri(tempfilt_yn2) 1\n'
        dg_temp += '\n'   
        dg_temp += '# Add temporal derivative (EV 2)\n'
        dg_temp +- 'set fmri(deriv_yn2) 0\n'
        
        evnum  = 0
        for ev in ev_target[0]:
            evnum += 1
            parfsf += '# EV ' + str(evnum) + ' title\n'
            parfsf += 'set fmri(evtitle' + str(evnum) + ') "' + ev.split('/')[-1][:-11] + '_ID"\n'
            parfsf += id_temp.replace('1)',''.join([str(evnum),')']))
            parfsf += '\n'
            parfsf += '# Custom EV file (EV ' + str(evnum) + ')\n'
            parfsf += 'set fmri(custom' + str(evnum) + ') "' + ev + '"\n'
            parfsf += '\n'
            parfsf += '# Gamma sigma (EV ' + str(evnum) + ')\n'
            parfsf += 'set fmri(gammasigma' + str(evnum) + ') .5\n'
            parfsf += '\n'
            parfsf += '# Gamma delay (EV ' + str(evnum) + ')\n'
            parfsf += 'set fmri(gammadelay' + str(evnum) + ') 2\n'
            parfsf += '\n'
            for j in range(2*len(ev_target) + 1):
                parfsf += '# Orthogonalise EV ' + str(evnum) + ' wrt EV ' + str(j) + '\n'
                parfsf += 'set fmri(ortho' + str(evnum) + ') ' + str(j) + '\n'
            parfsf += '\n'
            evnum += 1
            parfsf += '# EV ' + str(evnum) + ' title\n'
            parfsf += 'set fmri(evtitle' + str(evnum) + ') "' + ev.split('/')[-1][:-11] + '_DG"\n'
            parfsf += dg_temp.replace('2)',''.join([str(evnum),')']))
            parfsf += '\n'
            parfsf += '# Custom EV file (EV ' + str(evnum) + ')\n'
            parfsf += 'set fmri(custom' + str(evnum) + ') "' + ev + '"\n'
            parfsf += '\n'
            for j in range(2*len(ev_target) + 1):
                parfsf += '# Orthogonalise EV ' + str(evnum) + ' wrt EV ' + str(j) + '\n'
                parfsf += 'set fmri(ortho' + str(evnum) + ') ' + str(j) + '\n'
            parfsf += '\n'
        parfsf += '# Contrast & F-tests mode\n'
        parfsf += '# real : control real EVs\n'
        parfsf += '# orig : control original EVs\n'
        parfsf += 'set fmri(con_mode_old) orig\n'
        parfsf += 'set fmri(con_mode) orig\n'
        parfsf += '\n'
        
        evnum = 0
        for ev in ev_target:
            evnum += 1
            parfsf += '# Display images for contrast_real ' + str(evnum) + '\n'
            parfsf += 'set fmri(conpic_real.' + str(evnum) + ') ' + str(evnum) + '\n'
            parfsf += '\n'
            parfsf += '# Title for contrast_real ' + str(evnum) + '\n'
            parfsf += 'set fmri(conname_real.' + str(evnum) + ') "' + ev.split('/')[-1][:-11] + '_ID"\n'
            parfsf += '\n'
            for j in range(1,2*len(ev_target) + 1):
                parfsf += '# Real contrast_real vector ' + str(evnum) + ' element ' + str(j) + '\n'
                if evnum == j:
                    parfsf += 'set fmri(con_real' + str(evnum) + '.' + str(j) + ') -1.0\n'
                else:
                    parfsf += 'set fmri(con_real' + str(evnum) + '.' + str(j) + ') 0\n'
            parfsf += '\n'
            evnum += 1
            parfsf += '# Display images for contrast_real ' + str(evnum) + '\n'
            parfsf += 'set fmri(conpic_real.' + str(evnum) + ') ' + str(evnum) + '\n'
            parfsf += '\n'
            parfsf += '# Title for contrast_real ' + str(evnum) + '\n'
            parfsf += 'set fmri(conname_real.' + str(evnum) + ') "' + ev.split('/')[-1][:-11] + '_DG"\n'
            parfsf += '\n'
            for j in range(1,2*len(ev_target) + 1):
                parfsf += '# Real contrast_real vector ' + str(evnum) + ' element ' + str(j) + '\n'
                if evnum == j:
                    parfsf += 'set fmri(con_real' + str(evnum) + '.' + str(j) + ') 1.0\n'
                else:
                    parfsf += 'set fmri(con_real' + str(evnum) + '.' + str(j) + ') 0\n'
            parfsf += '\n'
            
        evnum = 0
        for ev in ev_target:
            evnum += 1
            parfsf += '# Display images for contrast_orig ' + str(evnum) + '\n'
            parfsf += 'set fmri(conpic_orig.' + str(evnum) + ') ' + str(evnum) + '\n'
            parfsf += '\n'
            parfsf += '# Title for contrast_orig ' + str(evnum) + '\n'
            parfsf += 'set fmri(conname_orig.' + str(evnum) + ') "' + ev.split('/')[-1][:-11] + '_ID"\n'
            parfsf += '\n'
            for j in range(1,2*len(ev_target) + 1):
                parfsf += '# Real contrast_orig vector ' + str(evnum) + ' element ' + str(j) + '\n'
                if evnum == j:
                    parfsf += 'set fmri(con_orig' + str(evnum) + '.' + str(j) + ') -1.0\n'
                else:
                    parfsf += 'set fmri(con_orig' + str(evnum) + '.' + str(j) + ') 0\n'
            parfsf += '\n'
            evnum += 1
            parfsf += '# Display images for contrast_orig ' + str(evnum) + '\n'
            parfsf += 'set fmri(conpic_orig.' + str(evnum) + ') ' + str(evnum) + '\n'
            parfsf += '\n'
            parfsf += '# Title for contrast_orig ' + str(evnum) + '\n'
            parfsf += 'set fmri(conname_orig.' + str(evnum) + ') "' + ev.split('/')[-1][:-11] + '_DG"\n'
            parfsf += '\n'
            for j in range(1,2*len(ev_target) + 1):
                parfsf += '# Real contrast_orig vector ' + str(evnum) + ' element ' + str(j) + '\n'
                if evnum == j:
                    parfsf += 'set fmri(con_orig' + str(evnum) + '.' + str(j) + ') 1.0\n'
                else:
                    parfsf += 'set fmri(con_orig' + str(evnum) + '.' + str(j) + ') 0\n'
            parfsf += '\n'    
        
        parfsf += '# Contrast masking - use >0 instead of thresholding?\n'
        parfsf += 'set fmri(conmask_zerothresh_yn) 0\n'
        evnum = 0
        for j in range(1,2*len(ev_target) + 1):
            for k in range(1,2*len(ev_target) + 1):
                if j != k:
                    parfsf += '# Mask real contrast/F-test ' + str(j) + ' with real contrast/F-test ' + str(k) + '?\n'
                    parfsf += 'set fmri(conmask' + str(j) + '_' + str(k) + ') 0\n'
        parfsf += '\n'
        parfsf += '# Do contrast masking at all?\n'
        parfsf += 'set fmri(conmask1_1) 0\n'
        parfsf += '\n'
        parfsf += '##########################################################\n'
        parfsf += "# Now options that don't appear in the GUI\n"
        parfsf += '\n'
        parfsf += '# Alternative (to BETting) mask image\n'
        parfsf += 'set fmri(alternative_mask) ""\n'
        parfsf += '\n'
        parfsf += '# Initial structural space registration initialisation transform\n'
        parfsf += 'set fmri(init_initial_highres) ""\n'
        parfsf += '\n'
        parfsf += '# Structural space registration initialisation transform\n'
        parfsf += 'set fmri(init_highres) ""\n'
        parfsf += '\n'
        parfsf += '# Standard space registration initialisation transform\n'
        parfsf += 'set fmri(init_standard) ""\n'
        parfsf += '\n'
        parfsf += '# For full FEAT analysis: overwrite existing .feat output dir?\n'
        parfsf += 'set fmri(overwrite_yn) 0\n'
        
        open("".join([FSFDIR,'design.fsf']), 'w').write(parfsf)
        parfsf = "".join([FSFDIR,'design.fsf'])
        
        if mp and not os.path.isdir(''.join([pardir,'fMRI_Analyses/Conceptual_Change_ID/',word.capitalize().replace('_target',''),str(runnum)])):
            #Run analysis                      
            analysis = Workflow(name = "Target_GLM_Level1")
            data = Node(ID(fields=['func','mp','ev_target','target_contrasts',
                                   'func_trim_point','mask','numevs'],
                           mandatory_inputs=False),name='Data')
            data.inputs.func = func
            data.inputs.mp = mp
            data.inputs.ev_target = [ev_target]
            #data.inputs.ev_fixation_category = [ev_fixation_category]
            data.inputs.target_contrasts = target_contrasts
            #data.inputs.fixation_category_contrasts = fixation_category_contrasts
            data.inputs.func_trim_point = func_trim_point
            data.inputs.mask = '/gpfs/group/pul8/default/read/MNI152_T1_3mm3mm4mm_brain_mask.nii.gz'
            data.inputs.numevs = 2*len(ev_target)
            
            #Use the DataSink function to store all outputs
            datasink = Node(nio.DataSink(), name= 'Output')
            datasink.inputs.base_directory = pardir + 'fMRI_Analyses/Conceptual_Change/' + word.capitalize().replace('_target','') + str(runnum)
            
            #Now we will remove all nuisance signals in native space, before moving the
            #data to MNI space.
            id_feat = MapNode(Function(input_names=['parfsf','func','numevs'],
                                        output_names = ['copes','dof_file','param_estimates','residual4d','sigmasquareds','thresholdac','tstats','varcopes','zstats'],
                                        function = ID_Feat),
                                        name = "ID_Feat",
                                        iterfield = ['parfsf','rest'])
            analysis.connect(data,'func',id_feat,'func')
            analysis.connect(data,'parfsf',id_feat,'parfsf')
            analysis.connect(data,'numevs',id_feat,'parevs')
            analysis.connect(id_feat,'copes',datasink,'Copes')
            analysis.connect(id_feat,'dof_file',datasink,'DOF')
            #analysis.connect(pnm_feat,'fstats',datasink,'Parametric_Fstats')
            analysis.connect(id_feat,'param_estimates',datasink,'Param_Estimates')
            analysis.connect(id_feat,'residual4d',datasink,'Residual4D')
            analysis.connect(id_feat,'sigmasquareds',datasink,'Sigma_Squareds')
            analysis.connect(id_feat,'thresholdac',datasink,'AC_Params')
            analysis.connect(id_feat,'tstats',datasink,'Tstats')
            analysis.connect(id_feat,'varcopes',datasink,'Varcopes')
            #analysis.connect(pnm_feat,'zfstats',datasink,'Parametric_Z_Fstats')
            analysis.connect(id_feat,'zstats',datasink,'Z_Tstats')
            
            analysis.run()   
    '''
    analysis2 = Workflow(name = "Target_GLM_Level2")
    data = Node(ID(fields=['func','mp','ev_target','target_contrasts',
                           'func_trim_point','mask','numevs'],
                   mandatory_inputs=False),name='Data')
    data.inputs.func = func
    data.inputs.mp = mp
    data.inputs.ev_target = [ev_target]
    #data.inputs.ev_fixation_category = [ev_fixation_category]
    data.inputs.target_contrasts = target_contrasts
    #data.inputs.fixation_category_contrasts = fixation_category_contrasts
    data.inputs.func_trim_point = func_trim_point
    data.inputs.mask = '/gpfs/group/pul8/default/read/MNI152_T1_3mm3mm4mm_brain_mask.nii.gz'
    data.inputs.numevs = 2*len(ev_target)
    
    #Use the DataSink function to store all outputs
    datasink = Node(nio.DataSink(), name= 'Output')
    datasink.inputs.base_directory = pardir + 'fMRI_Analyses/Conceptual_Change/' + word.capitalize().replace('_target','') + str(runnum)        
    concrete_level2model = Node(fsl.model.L2Model(num_copes = len(mp)),
                                         name = 'Concrete_L2Model')
    #analysis.connect(data,'numcopes',concrete_level2model,'num_copes')
    analysis2.connect(concrete_level2model,'design_con',datasink,'Concrete_L2_TCons')
    analysis2.connect(concrete_level2model,'design_mat',datasink,'Concrete_L2_Design')
    analysis2.connect(concrete_level2model,'design_grp',datasink,'Concrete_L2_Group')
    
    
    concrete_copemerge = MapNode(interface=fsl.Merge(dimension='t'),
                              iterfield=['in_files'],
                              name="Concrete_Cope_Merge")
    analysis2.connect(concrete_filmgls,('copes',sort_copes),concrete_copemerge,'in_files')
    analysis2.connect(concrete_copemerge,'merged_file',datasink,'Concrete_Merged_Copes')
    
    concrete_varcopemerge = MapNode(interface=fsl.Merge(dimension='t'),
                           iterfield=['in_files'],
                           name="Concrete_Varcope_Merge")
    analysis2.connect(concrete_filmgls,('varcopes',sort_copes),concrete_varcopemerge,'in_files')
    analysis2.connect(concrete_varcopemerge,'merged_file',datasink,'Concrete_Merged_Varcopes')
    
    concrete_flameo = MapNode(interface=fsl.model.FLAMEO(run_mode = 'fe'),
                                                           name = 'Concrete_Fixed_Effects',
                                                           iterfield = ['cope_file','var_cope_file'])#],'dof_var_cope_file'])  
    analysis2.connect(concrete_copemerge,'merged_file',concrete_flameo,'cope_file')
    analysis2.connect(concrete_varcopemerge,'merged_file',concrete_flameo,'var_cope_file')
    analysis2.connect(concrete_level2model,'design_con',concrete_flameo,'t_con_file')
    analysis2.connect(concrete_level2model,'design_mat',concrete_flameo,'design_file')
    analysis2.connect(concrete_level2model,'design_grp',concrete_flameo,'cov_split_file')
    analysis2.connect(data,'mask',concrete_flameo,'mask_file')
    #analysis.connect(parametric_filmgls,'dof_file',parametric_flameo,'dof_var_cope_file')
    analysis2.connect(concrete_flameo,'copes',datasink,'Concrete_Fixed_Effects_Copes')
    analysis2.connect(concrete_flameo,'res4d',datasink,'Concrete_Fixed_Effects_Residuals')
    analysis2.connect(concrete_flameo,'tdof',datasink,'Concrete_Fixed_Effects_TDOF')
    analysis2.connect(concrete_flameo,'tstats',datasink,'Concrete_Fixed_Effects_Tstats')
    analysis2.connect(concrete_flameo,'var_copes',datasink,'Concrete_Fixed_Effects_Varcopes')
    analysis2.connect(concrete_flameo,'zstats',datasink,'Concrete_Fixed_Effects_Z_Tstats')
    analysis2.run()'''