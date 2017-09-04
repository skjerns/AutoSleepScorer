# -*- coding: utf-8 -*-
"""
This is python 3 code
main script for training/classifying
"""
if not '__file__' in vars(): __file__= u'C:/Users/Simon/dropbox/Uni/Masterthesis/AutoSleepScorer/main.py'
import os
import gc; gc.collect()
import tools

import dill as pickle
import scipy
import numpy as np
import sleeploader
import matplotlib; 
matplotlib.rcParams['figure.figsize'] = (10, 3)
matplotlib.use('Agg')
if __name__ == '__main__': # outsource TF imports for MP
    import keras
    import models
    import keras_utils
    from keras_utils import cv

np.random.seed(42)
#%%
if __name__ == '__main__':
    try:
        with open('count') as f:
            counter = int(f.read())
    except IOError:
        print('No previous experiment?')
        counter = 0
         
    with open('count', 'w') as f:
      f.write(str(counter+1))        
    
    
    if os.name == 'posix':
        datadir  = './'
    
    else:
        datadir = 'c:\\sleep\\data\\'
    #    datadir = 'C:\\sleep\\vinc\\brainvision\\correct\\'
        datadir = 'c:\\sleep\\vinc\\'
    
    def load_data():
        global sleep
        global data
        sleep = sleeploader.SleepDataset(datadir)
        if 'data' in vars():  
            del data; 
            gc.collect()
        sleep.load_object()
    
        data, target, groups = sleep.get_all_data(groups=True)
        data    = tools.normalize(data)
        
        target[target==4] = 3
        target[target==5] = 4
        target[target==8] = 0
        target = keras.utils.to_categorical(target)

        return data, target, groups
        
    data,target,groups = load_data()
    #%%
#    import stopping
    print('Extracting features')

#    feats_eeg = tools.feat_eeg(data[:,:,0])
#    feats_emg = tools.feat_emg(data[:,:,1])
#    feats_eog = tools.feat_eog(data[:,:,2])
#
#    np.save('target.npy')
#    np.save('groups.npy')
#    np.save('feats_eeg.npy')
#    np.save('feats_eog.npy')
#    np.save('feats_emg.npy')

#    feats_all = np.hstack([feats_eeg, feats_emg, feats_eog])
#%%
#    print("starting")
#    comment = 'testing_electrodes for feat'
#    print(comment)
#    plot = False
#    ##%% 
#    epochs = 250
#    batch_size = 512
#
    results = dict()

#
#    import stopping
#    with open('results_electrodes.pkl', 'wb') as f:
#                pickle.dump(results, f)
    ###%% 
    epochs = 250
    batch_size = 256
    #
    cropsize = 2800
#    r = cv(data[:,:,0:1],   target, groups, models.cnn3adam_filter_morel2, epochs=epochs, name = 'eeg', stop_after=15, counter=counter,batch_size=batch_size, cropsize=cropsize)
#    results.update(r)
##    r = cv(data[:,:,[0,1]], target, groups, models.cnn3adam_filter_morel2, epochs=epochs, name = 'eeg+emg', stop_after=15, counter=counter,batch_size=batch_size, cropsize=cropsize) 
#    results.update(r)
#    r = cv(data[:,:,[0,2]], target, groups, models.cnn3adam_filter_morel2, epochs=epochs, name = 'eeg+eog', stop_after=15, counter=counter,batch_size=batch_size, cropsize=cropsize)  
#    results.update(r)
    r = cv(data[:,:,:],     target, groups, models.cnn3dilated, epochs=epochs, name = 'all', stop_after=15, counter=counter,batch_size=batch_size, cropsize=cropsize, balanced=True) 
#    results.update(r)
#    with open('results_electrodes_morel2.pkl', 'wb') as f:
#                pickle.dump(results, f)
    
    

