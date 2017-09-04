# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:02:04 2017

@author: Simon
"""
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import re
import time
from tools import plot_confusion_matrix



def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]   

#%%

#l=a['feat_eeg']
#val_acc = [y[0] for y in [x for x in l]]
#val_f1 = [y[1] for y in [x for x in l]]
#test_acc = [y[2] for y in [x for x in l]]
#test_f1 = [y[3] for y in [x for x in l]]
#
#val = np.vstack([val_acc, val_f1]).T
#test = np.vstack([test_acc, test_f1]).T

#a   = pickle.load(open('./results_dataset_feat_edfx','rb'))


names = sorted(a.keys(), key=natural_key)
res = [[a[key]] for key in names]



i=0
all_scores = list()
for exp in res:
    print (exp)
    scores = list()
    for fold in exp[0]:
        scores.append(fold[i])
    all_scores.append(scores)
s1 = np.vstack(all_scores).T
i=1
all_scores = list()
for exp in res:
    print (exp)
    scores = list()
    for fold in exp[0]:
        scores.append(fold[i])
    all_scores.append(scores)
s2 = np.vstack(all_scores).T
i=2
all_scores = list()
for exp in res:
    print (exp)
    scores = list()
    for fold in exp[0]:
        scores.append(fold[i])
    all_scores.append(scores)
s3 = np.vstack(all_scores).T
i=3
all_scores = list()
for exp in res:
    print (exp)
    scores = list()
    for fold in exp[0]:
        scores.append(fold[i])
    all_scores.append(scores)
s4 = np.vstack(all_scores).T

copypasta = []
copypasta.extend(['\t'.join([i for i in names])])
copypasta.extend(['\t'.join([str(x) for x in i]) for i in s1])
copypasta.append('')
copypasta.append('')
copypasta.append('')
copypasta.extend(['\t'.join([str(x) for x in i]) for i in s2])
copypasta.append('')
copypasta.append('')
copypasta.append('')
copypasta.extend(['\t'.join([str(x) for x in i]) for i in s3])
copypasta.append('')
copypasta.append('')
copypasta.append('')
copypasta.extend(['\t'.join([str(x) for x in i]) for i in s4])
copypasta.append('')
copypasta.append('')
copypasta.append('')


stop
#%%
results = dill.load('results_transfer_cshs50_all')
s = []
for exp in sorted(results):
    print (exp)
    res = results[exp]
    s.extend([x for x in res[:4]])
    s.extend([''])

