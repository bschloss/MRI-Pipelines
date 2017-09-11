# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:07:40 2016

@author: bschloss
"""
import argparse as ap
import numpy as np
import os
import exceptions
import subprocess as sp

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

ev_dir = pardir+ 'EV/'
cot_fix_files = [ev_dir + 'Run' + str(i) +'/Content_or_Target_Word_with_Words.run00' + str(i) + '.txt' for i in range(1,6)] 
cot_fix = [[] for i in range(1,6)]
end_vols = [ev_dir + 'Run' + str(i) + '/text_end_vol_num.run00' + str(i) + '.txt' for i in range(1,6)]
words = {}

run = 0
for f in cot_fix_files:
    end_vol = int(open(end_vols[run],'r').read().rstrip(' \n '))
    run+=1
    for line in open(f,'r').readlines():
        w = line.split()[1]
        onset = float(line.split()[2])
        onset_vol =  int(np.round(onset/.400))
        if w not in words.keys():
            words[w] = {1: (run,str(onset_vol),str(min(onset_vol+20,end_vol)))}
        else:
            words[w][max(words[w].keys())+1] = (run,str(onset_vol),str(min(onset_vol+20,end_vol)))
            
for w in words.keys():
    for num in words[w].keys():
        d = pardir + 'Word_Vols/' + w + '/' + str(num) + '/'
        try:
            os.makedirs(d)
        except:
            exceptions.OSError
        source = pardir + 'fMRI_Preprocessing/Highpass/_HighPass_Filter' + str(int(words[w][num][0]) - 1) + '/'
        source += os.listdir(source)[0]
        output = d + words[w][num][1] + '-' + str(int(words[w][num][2])-1)
        extract = sp.Popen(['fslroi',source,output,words[w][num][1],str(int(words[w][num][2])-int(words[w][num][1]))])
        sp.Popen.wait(extract)

            
                   
        
