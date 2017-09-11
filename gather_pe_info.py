# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 22:25:16 2017

@author: bschloss

This script makes a table of the phase encoding information in the dicom header
for each participant for the relevant images and stores it in a tab delimited
text file
"""

import os
import exceptions 

def get_direction(nifti):
    import nibabel as nib
    import exceptions
    header = nib.load(nifti).header
    try:
        direction = str(header['descrip']).split(';')[2][-1]
    except:
        exceptions.IndexError
        direction = 'Missing'
    if direction == '+':
        direction = "'+"
    return direction

def get_orientation(directory):
    import os
    import dicom
    f1 = directory + '/' + os.listdir(directory)[0]
    orientation = dicom.read_file(f1,force=True).InPlanePhaseEncodingDirection
    return orientation
    
#List of participant directores
pardirs = ['201','002','003','004','105','006','107','008','009','110',         
	   '011','012','013','214','015','016','017','018','019','020',
	   '021','122','023','024','025','026','027','028','029','030',
	   '031','132','033','034','035','036','037','038','039','040',
	   '041','042','043','044','045','046','047','048','049','050']
        #'118','140']

table = ''
  
#Iterate over participant folders
for par in pardirs:
    raw = '/'.join(['/gpfs/group/pul8/default/read',par,'Raw'])
    os.chdir(raw)
    
    #Check how the dicoms were converted and organized
    if 'ser1' in os.listdir(raw):
        #Get the series numbers for runs 1 - 5
        func1 = '/gpfs/group/pul8/default/read/' + par + '/Func/Run1/'
        try:
            func1dir = get_direction(''.join([func1,os.listdir(func1)[0]]))
            func1 = 'ser' + str(int(os.listdir(func1)[0][-14:-11]))
        except:
            exceptions.NameError
            del func1
            
        func2 = '/gpfs/group/pul8/default/read/' + par + '/Func/Run2/'
        try:
            func2dir = get_direction(''.join([func2,os.listdir(func2)[0]]))
            func2 = 'ser' + str(int(os.listdir(func2)[0][-14:-11]))
        except:
            exceptions.NameError
            del func2
            
        func3 = '/gpfs/group/pul8/default/read/' + par + '/Func/Run3/'
        try:
            func3dir = get_direction(''.join([func3,os.listdir(func3)[0]]))
            func3 = 'ser' + str(int(os.listdir(func3)[0][-14:-11]))
        except:
            exceptions.NameError
            del func3
        
        func4 = '/gpfs/group/pul8/default/read/' + par + '/Func/Run4/'
        try:
            func4dir = get_direction(''.join([func4,os.listdir(func4)[0]]))
            func4 = 'ser' + str(int(os.listdir(func4)[0][-14:-11]))
        except:
            exceptions.NameError
            del func4
        
        func5 = '/gpfs/group/pul8/default/read/' + par + '/Func/Run5/'
        try:
            func5dir = get_direction(''.join([func5,os.listdir(func5)[0]]))
            func5 = 'ser' + str(int(os.listdir(func5)[0][-14:-11]))
        except:
            exceptions.NameError
            del func5
        
        #Get the series numbers for SE-AP and SE-PA
        ses = os.listdir('/'.join(['/gpfs/group/pul8/default/read',par,'Fieldmap']))
        for se in ses:
            if 'Grappa' not in se:
                if 'AP' in se:
                    seapdir = get_direction('/'.join(['/gpfs/group/pul8/default/read',par,'Fieldmap',se]))
                    seap = 'ser' + str(int(se[-14:-11]))
                if 'PA' in se:
                    sepadir = get_direction('/'.join(['/gpfs/group/pul8/default/read',par,'Fieldmap',se]))
                    sepa = 'ser' + str(int(se[-14:-11]))
        
        #Get the series numbers for the DTI and DTI PA scans
        dti = os.listdir('/'.join(['/gpfs/group/pul8/default/read',par,'DTI']))
        for img in dti:
            if '.nii.gz' in img:
                if 'PA' in img:
                    dtipadir = get_direction('/'.join(['/gpfs/group/pul8/default/read',par,'DTI',img]))
                    dtipa = 'ser' + str(int(img[-15:-12]))
                else:
                    dtiapdir = get_direction('/'.join(['/gpfs/group/pul8/default/read',par,'DTI',img]))
                    dtiap = 'ser' + str(int(img[-15:-12]))
    else:
        for d in os.listdir('/'.join(['/gpfs/group/pul8/default/read',par,'Raw'])):
            if "RUN_1_" in d and "SBREF" not in d:
                func1 = d
                func1dir = '/gpfs/group/pul8/default/read/' + par + '/Func/Run1/'
                func1dir = get_direction(''.join([func1dir,os.listdir(func1dir)[0]]))
            if "RUN_2_" in d and "SBREF" not in d:
                func2 = d
                func2dir = '/gpfs/group/pul8/default/read/' + par + '/Func/Run2/'
                func2dir = get_direction(''.join([func2dir,os.listdir(func2dir)[0]]))
            if "RUN_3_" in d and "SBREF" not in d:
                func3 = d
                func3dir = '/gpfs/group/pul8/default/read/' + par + '/Func/Run3/'
                func3dir = get_direction(''.join([func3dir,os.listdir(func3dir)[0]]))
            if "RUN_4_" in d and "SBREF" not in d:
                func4 = d
                func4dir = '/gpfs/group/pul8/default/read/' + par + '/Func/Run4/'
                func4dir = get_direction(''.join([func4dir,os.listdir(func4dir)[0]]))
            if "RUN_5_" in d and "SBREF" not in d:
                func5 = d
                func5dir = '/gpfs/group/pul8/default/read/' + par + '/Func/Run5/'
                func5dir = get_direction(''.join([func5dir,os.listdir(func5dir)[0]]))    
            ses = os.listdir('/'.join(['/gpfs/group/pul8/default/read',par,'Fieldmap']))
            for se in ses:
                if 'Grappa' not in se:
                    if 'AP' in se:
                        seapdir = get_direction('/'.join(['/gpfs/group/pul8/default/read',par,'Fieldmap',se]))
                    if 'PA' in se:
                        sepadir = get_direction('/'.join(['/gpfs/group/pul8/default/read',par,'Fieldmap',se]))
            if "SPINECHOFIELDMAP_AP_" in d and "SBREF" not in d:
                seap = d
            if "SPINECHOFIELDMAP_PA_" in d and "SBREF" not in d:
                sepa = d
            
            #Get the series numbers for the DTI and DTI PA scans
            dti = os.listdir('/'.join(['/gpfs/group/pul8/default/read',par,'DTI']))
            for img in dti:
                if '.nii.gz' in img:
                    if 'PA' in img:
                        dtipadir = get_direction('/'.join(['/gpfs/group/pul8/default/read',par,'DTI',img]))
                    else:
                        dtiapdir = get_direction('/'.join(['/gpfs/group/pul8/default/read',par,'DTI',img]))
            if "CMRR_MBEP2D_DIFF" in d:
                if "P-A" not in d:
                    dtiap = d
                else:
                    dtipa = d
                    
    if 'func1' in locals():
        table += par + '\t\t' + 'Run1\t' + get_orientation(func1) + '\t' + func1dir + '\n'
        del func1
    else:
        table += par + '\t\t' + 'Run1\tMissing\tMissing\n'
        
    if 'func2' in locals():
        table += '\t\t' + 'Run2\t' + get_orientation(func2) + '\t' + func2dir + '\n'
        del func2
    else:
        table += '\t\tRun2\tMissing\tMissing\n'  
        
    if 'func3' in locals():
        table += '\t\t' + 'Run3\t' + get_orientation(func3) + '\t' + func3dir +'\n'
        del func3
    else:
        table += '\t\tRun3\tMissing\tMissing\n'
        
    if 'func4' in locals():
        table += '\t\t' + 'Run4\t' + get_orientation(func4) + '\t' + func4dir + '\n'
        del func4
    else:
        table += '\t\tRun4\tMissing\tMissing\n'
        
    if 'func5' in locals():
        table += '\t\t' + 'Run5\t' + get_orientation(func5) + '\t' + func5dir +'\n'
        del func5
    else:
        table += '\t\tRun5\tMissing\tMissing\n'
    
    if 'seap' in locals():
        table += '\t\t' + 'SE AP\t' + get_orientation(seap) + '\t' + seapdir +'\n'
        del seap
    else:
        table += '\t\tSE AP\tMissing\tMissing\n'
        
    if 'sepa' in locals():
        table += '\t\t' + 'SE PA\t' + get_orientation(sepa) + '\t' + sepadir +'\n'
        del sepa
    else:
        table += '\t\tSE PA\tMissing\tMissing\n'
        
    if 'dtiap' in locals():
        table += '\t\t' + 'DTI AP\t' + get_orientation(dtiap) + '\t' + dtiapdir +'\n'
        del dtiap
    else:
        table += '\t\tDTI AP\tMissing\tMissing\n'
        
    if 'dtipa' in locals():
        table += '\t\t' + 'DTI PA\t' + get_orientation(dtipa) + '\t' + dtipadir +'\n'
        del dtipa
    else:
        table += '\t\tDTI PA\tMissing\tMissing\n'      
          
open('/gpfs/group/pul8/default/read/Hershey_PE_direction_info.txt','w').write(table)          
