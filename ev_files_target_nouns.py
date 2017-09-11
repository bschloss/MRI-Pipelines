# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:01:02 2016

@author: bschloss
"""
import argparse as ap
import string
import shutil

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
        
    target = ['gps',
              'mars',
              'axial',
              'battery',
              'circuit',
              'combination',
              'current',
              'distance',
              'earth',
              'electric',
              'electron',
              'gravity',
              'item',
              'lightbulb',
              'live',
              'location',
              'number',
              'ocean',
              'oil',
              'orbit',
              'order',
              'path',
              'permutation',
              'radio',
              'receiver',
              'rover',
              'safety',
              'satellite',
              'scientist',
              'select',
              'set',
              'signal',
              'source',
              'space',
              'spill',
              'temperature',
              'tilt',
              'time',
              'wire']
            
    evs = {}
    evs['run001'] = {}
    evs['run002'] = {}
    evs['run003'] = {}
    evs['run004'] = {}
    evs['run005'] = {}
    
    for run in evs.keys():
        for word in target:
            evs[run]['_'.join([word,'Other_Content_or_Target'])] = ''
            evs[run]['_'.join([word,'target'])] = {}
            
    parevs = [pardir + 'EV/Run' + str(i) + '/Content_or_Target_Word_with_Words.run00' + str(i) + '.txt' for i in range(1,6)]
    pardum = [pardir + 'EV/Run' + str(i) + '/Dummy_Content_or_Target_with_Words.run00' + str(i) + '.txt' for i in range(1,6)]
    
    for tword in target:
        run_num = 0
        for ev in parevs:
            run_num += 1
            if not (pardir[-3:-1] == '21' and run_num == 4):
                run = 'run00' + str(run_num)
                data = open(ev,'r').readlines()
                for line in data:
                    word = charsonly(line.split()[1])
                    if word == tword:
                        evs[run]['_'.join([tword,'target'])][len(evs[run]['_'.join([tword,'target'])])+1] = '\t'.join(line.split()[2:]) + '\n'
                    else:
                        evs[run]['_'.join([tword,'Other_Content_or_Target'])] += '\t'.join(line.split()[2:]) + '\n'
        run_num = 0
	for dum in pardum:
	    run_num += 1
	    if not (pardir[-3:-1] == '21' and run_num == 4):
		run = 'run00' + str(run_num)
		data = open(dum,'r').readlines()
		for line in data:
		    word = charsonly(line.split()[1])
		    if word == tword:
			evs[run]['_'.join([tword,'target'])][len(evs[run]['_'.join([tword,'target'])])+1] = '\t'.join(line.split()[2:]) + '\n'
		    else:
			evs[run]['_'.join([tword,'Other_Content_or_Target'])] += '\t'.join(line.split()[2:]) + '\n'

    for run in evs.keys():
        if not (pardir[-3:-1] == '21' and run == 'run004'):
            no_interest_src = pardir + 'EV/Run' + str(run[-1]) + '/No_Interest.run00' + str(run[-1]) + '.txt'
            no_interest_dest = no_interest_src.replace('EV/','EV2/')
            shutil.copy(no_interest_src,no_interest_dest)
            text_end_src = pardir + 'EV/Run' + str(run[-1]) + '/text_end_vol_num.run00' + str(run[-1]) + '.txt'
            text_end_dest = text_end_src.replace('EV/','EV2/')
            shutil.copy(text_end_src,text_end_dest)
            for ev in evs[run].keys():
                if 'Other_Content_or_Target' in ev:
                    if evs[run][ev] == '':
                        evs[run][ev] = '0\t0\t0'
                    evs[run][ev] = evs[run][ev].rstrip('\n')
		    evs[run][ev] = sorted([line.split('\t') for line in evs[run][ev].split('\n')],key=lambda x: float(x[0]))
		    evs[run][ev] = '\n'.join(['\t'.join(line) for line in evs[run][ev]])
                    fname = pardir + 'EV2/Run' + str(run[-1]) + '/' + ev + '.' + run + '.txt'
                    open(fname,'w').write(evs[run][ev])
                else:
                    for i in range(1,len(evs[run][ev])+1):
                        #if evs[run][ev][i] == '':
                        #    evs[run][ev][i] = '0\t0\t0'
                        #evs[run][ev][i] = evs[run][ev][i].rstrip('\n')
                        fname = pardir + 'EV2/Run' + str(run[-1]) + '/' + ev + str(i) + '.' + run + '.txt'
                        open(fname,'w').write(evs[run][ev][i])

    
if __name__=='__main__': 
    main()      
