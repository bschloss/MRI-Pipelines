# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 21:05:15 2017

@author: bschloss

This script was created to organize a folder which contains all of the dicom 
files from a single run, and put them in separate folders based on the dicom
series number.
"""

import os
import shutil as sh
import dicom
import argparse as ap

#Get participant directory from command line
parser = ap.ArgumentParser(description='Organize Dicom Files into Folders by scan series number')
parser.add_argument('par', metavar="stanford", type=str,
                    help="Path to participant's directory")                                                  
args = parser.parse_args()
if len(args.par) == 1:
    par = '00' + args.par
elif len(args.par) == 2:
    par = '0' + args.par
else:
    par = args.par  
    
#Go to the directory where raw data files are
raw = '/'.join(['/gpfs/group/pul8/default/read',par,'Raw'])                 
os.chdir(raw)                                                               

if os.listdir(raw)[0][-3:].lower() in ['ima','dcm','.v2'] or os.listdir(raw)[0][:2] == 'MR':
    #Collect all of the files that end in IMA or ima and make set of all the different series indices (indicated after the third period)
    series_ima = [str(int(num)) for num in set([f.split('.')[3] for f in os.listdir(raw) if f.split('.')[-1].lower() == 'ima'])]
    series_dcm = [str(int(num)) for num in set([f.split('_')[1] for f in os.listdir(raw) if f.split('.')[-1].lower() == 'dcm'])]
    series_v2 = [str(int(num)) for num in set([f.split('_')[1] for f in os.listdir(raw) if f.split('.')[-1].lower() == 'v2'])]
    series_MR = [dicom.read_file(f).SeriesNumber for f in os.listdir(raw) if f[:2] == 'MR']
    series = series_ima + series_dcm + series_v2 + series_MR
    series = set(series)
    
    #create subdirectories for the different dicom series
    for ser in series:
        newdir = ''.join(['ser',str(ser)])                                            
        os.mkdir(newdir)
    
    #Iterate over the files in the folder, check if it is an ima file, and,
    #if so, move it to the correct subfolder.                                                    
    for f in os.listdir(raw):
        if f.split('.')[-1].lower() == 'ima':
            src = f
            dest = 'ser' + str(int(f.split('.')[3])) + '/' + f
            sh.move(src,dest)
        if f.split('.')[-1].lower() == 'dcm' or f.split('.')[-1].lower() == 'v2':
            src = f
            dest = 'ser' + str(int(f.split('_')[1])) + '/' + f
            sh.move(src,dest)
        if f[:2] == 'MR':
            src = f 
            dest = 'ser' + str(dicom.read_file(f).SeriesNumber) + '/' + f
            sh.move(src,dest) 
