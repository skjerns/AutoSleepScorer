# -*- coding: utf-8 -*-
"""
This is python 3 code
main script for training/classifying
"""
import os
import gc; gc.collect()
import matplotlib
matplotlib.use('Agg')
import numpy as np
import keras
import tools
import scipy
import models
from sklearn.utils import shuffle
import pickle
import keras_utils
from copy import deepcopy
import telegram_send
from keras_utils import cv
import sleeploader
import matplotlib; matplotlib.rcParams['figure.figsize'] = (10, 3)
np.random.seed(42)
if __name__ == "__main__":
    try:
        with open('count') as f:
            counter = int(f.read())
    except IOError:
        print('No previous experiment?')
        counter = 0
         
    with open('count', 'w') as f:
      f.write(str(counter+1))        
    def load_data(dataset):
        sleep = sleeploader.SleepDataset(datadir)
        sleep.load_object(dataset)
        data, target, groups = sleep.get_all_data(groups=True)
        data = data[0:5000]
        target = target[0:5000]
        groups = groups[0:5000]
        data = tools.normalize(data, groups=groups)
        target[target==4] = 3
        target[target==5] = 4
        target[target>5] = 0
        target = keras.utils.to_categorical(target)
        print(np.unique(groups))
        return deepcopy(data), deepcopy(target), deepcopy(groups) 
    
    #%%
    datadir = './data' if os.name == 'posix' else 'd:\\sleep\\data'
    datasets = ['edfx', 'cshs50', 'cshs100', 'emsaad']
    data   = np.empty([0,3000,3])
    target = np.empty([0,5])
    groups = np.empty([0,])
    for dataset in datasets:
        gc.collect()
        sdata,starget,sgroups = load_data(dataset)
        data = np.append(data, sdata, axis=0)
        target = np.append(target, starget, axis=0)
        groups = np.append(groups, sgroups+len(np.unique(groups)), axis=0)
        
    data, target, groups = shuffle(data,target,groups)
    batch_size = 512
    epochs = 250
    name = 'multitrain_all-vinc'
    ###
    rnn = {'model':models.pure_rnn_do, 'layers': ['fc1'],  'seqlen':6,
           'epochs': 250,  'batch_size': 512,  'stop_after':15, 'balanced':False}
    print(rnn)
#    model = 'C:\\Users\\Simon\\dropbox\\Uni\\Masterthesis\\AutoSleepScorer\\weights\\balanced'
    model = models.cnn3adam_filter_l2
    r = keras_utils.cv (data, target, groups, model, rnn, name=name,
                             epochs=epochs, folds=5, batch_size=batch_size, counter=counter,
                             plot=True, stop_after=15, balanced=False, cropsize=2800)
    with open('results_multitrain_all-vinc.pkl', 'wb') as f:
                pickle.dump(r, f)
    telegram_send.send(['DONE: {} - {}\n{}'.format(os.path.basename(__file__), datasets,tools.print_string(r))])

    
    
