# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:01:02 2016

@author: bschloss
"""
import argparse as ap
import string
import shutil
import os

def charsonly(s):
	acceptable = string.ascii_letters 
    	new = ''
    	for char in s:  
		if char in acceptable:
			new += char
	return new.lower()
    
def main():  
    parser = ap.ArgumentParser(description='Create EV Files')
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
        
    conc =   ['axial',
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
        
    evs = {}
    evs['run001'] = {}
    evs['run002'] = {}
    evs['run003'] = {}
    evs['run004'] = {}
    evs['run005'] = {}
    
    for run in evs.keys():
        for word in conc:
            evs[run]['_'.join([word,'Other_Content_or_Target'])] = ''
            evs[run]['_'.join([word,'conc'])] = ''
    
    parevs = [pardir + 'EV/Run' + str(i) + '/Content_or_Target_Word_with_Words.run00' + str(i) + '.txt' for i in range(1,6)]
    pardummy = [pardir + 'EV/Run' + str(i) + '/Dummy_Content_or_Target_with_Words.run00' + str(i) + '.txt' for i in range(1,6)]
    for cword in conc:
        run_num = 0
        for ev in parevs:
            run_num += 1
            run = 'run00' + str(run_num)
            if not (pardir[-5:-1] == '/021' and run == 'run004') and not (pardir[-5:-1] == '2005' and run == 'run002') and not (pardir[-5:-1] in ['2006','2014'] and run == 'run005'):
                data = open(ev,'r').readlines()
                for line in data:
                    word = charsonly(line.split()[1])
                    if word == cword:
                        evs[run]['_'.join([cword,'conc'])] += line.split()[2] + '\t.400\t1.0\n'
                    else:
                        evs[run]['_'.join([cword,'Other_Content_or_Target'])] += line.split()[2] + '\t.400\t1.0\n'
        run_num = 0
	for ev in pardummy:
            run_num += 1
            run = 'run00' + str(run_num)
            if not (pardir[-5:-1] == '/021' and run == 'run004') and not (pardir[-5:-1] == '2005' and run == 'run002') and not (pardir[-5:-1] in ['2006','2014'] and run == 'run005'):
                data = open(ev,'r').readlines()
                for line in data:
                    word = charsonly(line.split()[1])
                    if word == cword:
                        evs[run]['_'.join([cword,'conc'])] += line.split()[2] + '\t.400\t1.0\n'
                    else:
                        evs[run]['_'.join([cword,'Other_Content_or_Target'])] += line.split()[2] + '\t.400\t1.0\n'
                            
    for run in evs.keys():
        if not (pardir[-5:-1] == '/021' and run == 'run004') and not (pardir[-5:-1] == '2005' and run == 'run002') and not (pardir[-5:-1] in ['2006','2014'] and run == 'run005'):
            no_interest_src = pardir + 'EV/Run' + str(run[-1]) + '/No_Interest.run00' + str(run[-1]) + '.txt'
            no_interest_dest = no_interest_src.replace('EV/','EV2/')
            shutil.copy(no_interest_src,no_interest_dest)
            text_end_src = pardir + 'EV/Run' + str(run[-1]) + '/text_end_vol_num.run00' + str(run[-1]) + '.txt'
            text_end_dest = text_end_src.replace('EV/','EV2/')
            shutil.copy(text_end_src,text_end_dest)
            for ev in evs[run].keys():
                if evs[run][ev].rstrip(' ') == '':
                    evs[run][ev] = '0\t0\t0'
                evs[run][ev] = evs[run][ev].rstrip('\n')
                fname = pardir + 'EV2/Run' + str(run[-1]) + '/' + ev + '.' + run + '.txt'
                open(fname,'w').write(evs[run][ev])
          
if __name__=='__main__': 
    main()      
