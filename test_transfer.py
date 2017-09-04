# -*- coding: utf-8 -*-
"""
This is python 3 code
main script for training/classifying
"""
import os
import gc; gc.collect()
import matplotlib;matplotlib.use('Agg')
if __name__=='__main__': import keras;import keras_utils
import tools
import dill as pickle
import scipy
from scipy.stats.mstats import zmap
import numpy as np
import sleeploader
from copy import deepcopy
import matplotlib
np.random.seed(42)

def create_object(datadir):
    sleep = sleeploader.SleepDataset(datadir)
    sleep.load()
    sleep.save_object()

def load_data(datadir, filename='sleepdata', norm = None):
    sleep = sleeploader.SleepDataset(datadir)

    if not sleep.load_object(filename): sleep.load()

    data, target, groups = sleep.get_all_data(groups=True)
    del sleep;gc.collect()

        
    target[target==4] = 3 # S3+S4 = SWS = 3
    target[target==5] = 4 # REM = 4
    target[target==8] = 0 # Movement = 0
    return data, target, groups
    

if __name__=='__main__':
    w_rnn = './weights/1180CNN+LSTM normalized-groupsfc1_3_0.878-0.819'
    cnn = keras.models.load_model(w_cnn)
    rnn = keras.models.load_model(w_rnn)
    datadirs = ['c:\\sleep\\cshs50',
                'c:\\sleep\\cshs100',
                'c:\\sleep\\edfx',
                'c:\\sleep\\vinc',
                'c:\\sleep\\emsa']
    
    targets = {}
    groups  = {}
    results = {}
    
    for datadir in datadirs:
        name = os.path.basename(datadir[:-1] if (datadir[-1] == '\\' or datadir[-1]=='/') else datadir)
        print('results for ', name)
        data, targets[name], groups[name] = load_data(datadir)
        targets[name] = deepcopy(targets[name])
        groups[name]  = deepcopy(groups[name])
        results[name] = keras_utils.test_data_cnn_rnn(data, targets[name], cnn, 'fc1', rnn, cropsize=2800, mode = 'preds')
        tools.plot_results_per_patient(results[name][0], results[name][2], groups[name], title='CNN '+ name)
        tools.plot_results_per_patient(results[name][1], results[name][3], groups[name][5:], title='CNN+LSTM '+ name)
        del data
        gc.collect()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
