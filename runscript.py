# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:02:08 2017

@author: Simon
"""

if __name__ == '__main__':
    import argparse
    import time
    import pickle
    import scipy
    import sleeploader
    import tools
    from keras_utils import cv
    import models
    import keras
    import matplotlib;matplotlib.use('Agg')
    import numpy as np
    import os
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('modus', metavar='N', type=str, help='which calculation to start')
    args = parser.parse_args()
    modus = args.modus
    try:
        with open('count') as f:
            counter = int(f.read())
    except Exception:
        print('No previous experiment?')
        counter = 0
    with open('count', 'w') as f:
      f.write(str(counter+1))  
      
    def load_data(datadir):
        sleep = sleeploader.SleepDataset(datadir)
        sleep.load_object()
        data, target, groups = sleep.get_all_data(groups=True)
        data    = tools.normalize(data , groups=groups)
#        target[target==5] = 4
        target[target==8] = 0
        target = keras.utils.to_categorical(target)
        return data, target, groups
        
    def cnn_eeg(c=0):
        r = cv(data[:,:,0:1], target, groups, models.cnn3adam_filter_l2, name = 'eeg', stop_after=15, counter=c, cropsize=2800, plot=plot)
        with open('results_electrodes_cnn_eeg.pkl', 'wb') as f: pickle.dump(r, f)
        
    def cnn_emg(c=0):
        r = cv(data[:,:,[0,1]], target, groups, models.cnn3adam_filter_l2, name = 'emg', stop_after=15, counter=c, cropsize=2800, plot=plot)
        with open('results_electrodes_cnn_emg.pkl', 'wb') as f: pickle.dump(r, f)  
        
    def cnn_eog(c=0):
        r = cv(data[:,:,[0,2]], target, groups, models.cnn3adam_filter_l2, name = 'eog', stop_after=15, counter=c, cropsize=2800, plot=plot)
        with open('results_electrodes_cnn_eog.pkl', 'wb') as f: pickle.dump(r, f)  
        
    def cnn_all(c=0):
        r = cv(data[:,:,:], target, groups, models.cnn3adam_filter_l2, name = 'all', stop_after=15, counter=c, cropsize=2800, plot=plot)
        with open('results_electrodes_cnn_all.pkl', 'wb') as f: pickle.dump(r, f)
        
    def feat_ann(c=0):
        batch_size =700
        feats_eeg = scipy.stats.zscore(tools.feat_eeg(data[:,:,0]))
        feats_emg = scipy.stats.zscore(tools.feat_emg(data[:,:,1]))

        feats_eog = scipy.stats.zscore(tools.feat_eog(data[:,:,2]))
        feats_all = np.hstack([feats_eeg, feats_emg, feats_eog])
        results = dict()
        r = cv(feats_eeg, target, groups, models.ann, name = 'eeg', stop_after=15,batch_size=batch_size, counter=c, plot=plot)
        results.update(r)
        r = cv(np.hstack([feats_eeg,feats_eog]), target, groups, models.ann, name = 'eeg+eog',batch_size=batch_size, stop_after=15, counter=c, plot=plot)  
        results.update(r)
        r = cv(np.hstack([feats_eeg,feats_emg]), target, groups, models.ann, name = 'eeg+emg',batch_size=batch_size, stop_after=15, counter=c, plot=plot) 
        results.update(r)
        r = cv(feats_all, target, groups, models.ann, name = 'all',batch_size=batch_size, stop_after=15, counter=c, plot=plot)
        results.update(r)
        with open('results_electrodes_feat.pkl', 'wb') as f:  pickle.dump(results, f)
        
    def feat_rnn(c=0):
        feats_eeg = scipy.stats.zscore(tools.feat_eeg(data[:,:,0]))
        feats_emg = scipy.stats.zscore(tools.feat_emg(data[:,:,1]))
        feats_eog = scipy.stats.zscore(tools.feat_eog(data[:,:,2]))
        feats_all = np.hstack([feats_eeg, feats_eog, feats_emg])
        feats_seq, targ_seq, groups_seq = tools.to_sequences(feats_all, target, groups=groups, seqlen=6, tolist=False)
        r = cv(feats_seq, targ_seq, groups_seq, models.pure_rnn_do, name = 'feat-rnn-all', stop_after=15, counter=c, plot=plot)
        with open('edfxresults_recurrent_feat.pkl', 'wb') as f:  pickle.dump(r, f)
    
    def lstm(c=0):
        batch_size = 256
        name = 'CNN+LSTM'
        rnn = {'model':models.pure_rnn_do, 'layers': ['fc1'],  'seqlen':6,
               'epochs': 250,  'batch_size': 512,  'stop_after':15, 'balanced':False}
        model = models.cnn3adam_filter_l2
        r = cv (data, target, groups, model, rnn, name=name, batch_size=batch_size, 
                counter=counter, plot=plot, stop_after=15, balanced=False, cropsize=2800)
        with open('results_recurrent_lstm.pkl', 'wb') as f: pickle.dump(r, f)
        
    datadir = '.' if os.name == 'posix' else 'c:/sleep/cshs50'
    data,target,groups = load_data(datadir)               
    start = time.time()  
    plot=True             
    if   modus == 'feat_ann':feat_ann(counter)
    elif modus == 'feat_rnn':feat_rnn(counter)
    elif modus == 'cnn_eeg':cnn_eeg(counter)
    elif modus == 'cnn_eog':cnn_eog(counter)
    elif modus == 'cnn_emg':cnn_emg(counter)
    elif modus == 'cnn_all':cnn_all(counter)
    elif modus == 'lstm':lstm(counter)
    else: print('Unknown mode selected')
    
    print('This took {:.1f} min'.format((time.time()-start)/60))