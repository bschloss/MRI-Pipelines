import os
import pickle

words = {}
for par in ['201','002','003','004','105','006','107','008','009','110',
	    '011','012','013','014','015','016','017','018','019','020',
	    '021','122','023','024','025','026','027','028','029','030',
	    '031','032','033','034','035','036','037','038','039','040',
	    '041','042']:
	wv_dir = '/gpfs/group/pul8/default/read/' + par + '/Word_Vols/'
	if os.path.isdir(wv_dir):
		for w in os.listdir(wv_dir):
			if w not in words.keys():
				words[w] = 0
			wdir = wv_dir + w + '/'
			if len(os.listdir(wdir))>3:
				words[w] += 1
swords = [(key,words[key]) for key in words.keys()]
swords.sort(key=lambda x: x[1])
pickle.dump(swords,open('/gpfs/group/pul8/default/read/words_freq_gt_4.pkl','wb'))
