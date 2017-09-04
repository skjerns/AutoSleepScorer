# -*- coding: utf-8 -*-
"""
This is python 3 code
main script for training/classifying
"""
if not '__file__' in vars(): __file__= u'C:/Users/Simon/dropbox/Uni/Masterthesis/AutoSleepScorer/main.py'
import os
import gc; gc.collect()
import matplotlib
matplotlib.use('Agg')
import numpy as np
import keras
import tools
import scipy
import models
import pickle
import keras_utils
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
    
    #%%
    target = keras.utils.to_categorical(np.load('target.npy'))
    groups = np.load('groups.npy')
    feats_eeg = np.load('feats_eeg.npy')
    feats_emg = np.load('feats_emg.npy')
    feats_eog = np.load('feats_eog.npy')

    feats_all = np.hstack([feats_eeg, feats_emg, feats_eog])
    
    
    batch_size = 512
    epochs = 250
    balanced = True
    results = {}
    r = cv(feats_eeg, target, groups, models.ann, epochs=epochs, name = 'eeg', stop_after=15, counter=counter,batch_size=batch_size, balanced=balanced) 
    results.update(r)
    r = cv(np.hstack([feats_eeg, feats_emg]), target, groups, models.ann, epochs=epochs, name = 'eeg+emg', stop_after=15, counter=counter,batch_size=batch_size, balanced=balanced) 
    results.update(r)
    r = cv(np.hstack([feats_eeg, feats_eog]), target, groups, models.ann, epochs=epochs, name = 'eeg+eog', stop_after=15, counter=counter,batch_size=batch_size, balanced=balanced) 
    results.update(r)
    r = cv(feats_all, target, groups, models.ann, epochs=epochs, name = 'all', stop_after=15, counter=counter,batch_size=batch_size, balanced=balanced) 
    results.update(r)
    with open('results_electrodes_feat_balanced.pkl', 'wb') as f:
        pickle.dump(results, f)



