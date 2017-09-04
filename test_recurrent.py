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
import telegram_send
import tools
import scipy
import models
from copy import deepcopy
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
    if os.name == 'posix':
        datadir  = '.'
    
    else:
    #    datadir = 'c:\\sleep\\data\\'
#        datadir = 'd:\\sleep\\vinc\\'
#        datadir = 'c:\\sleep\\emsa\\'
        datadir = 'c:\\sleep\\cshs50\\'
#        datadir = 'c:\\sleep\\edfx\\'

#
    
    def load_data(tsinalis=False):
        sleep = sleeploader.SleepDataset(datadir)
        gc.collect()
#        sleep.load()
    #    sleep.save_object()
        sleep.load_object()
        data, target, groups = sleep.get_all_data(groups=True)
    
        data = tools.normalize(data)
        target[target==4] = 3
        target[target==5] = 4
        target[target==8] = 0
        target = keras.utils.to_categorical(target)
        return deepcopy(data), deepcopy(target), deepcopy(groups)
        
    data,target,groups = load_data()
    datadir = 'c:\\sleep\\edfx\\'
#    data2,target2,groups2 = load_data()

    trans_tuple = [data,target,groups]
    #%%
    
#    print('Extracting features')
#    target = np.load('target.npy')
#    groups = np.load('groups.npy')
    feats_eeg = np.load('feats_eeg.npy')# tools.feat_eeg(data[:,:,0])
    feats_emg = np.load('feats_emg.npy')#tools.feat_emg(data[:,:,1])
    feats_eog = np.load('feats_eog.npy')#tools.feat_eog(data[:,:,2])
    feats_all = np.hstack([feats_eeg, feats_emg, feats_eog ])
    feats_all = scipy.stats.stats.zscore(feats_all)
#    # 
    if 'data' in vars():
        if np.sum(np.isnan(data)) or np.sum(np.isnan(data)):print('Warning! NaNs detected')
    #%%
    results = dict()

    comment = 'rnn_test'
    print(comment)
    
    print("starting at")
    #%%   model comparison
    #r = dict()
    #r['pure_rnn_5'] = cv(feats5, target5, groups5, models.pure_rnn, name='5',epochs=epochs, folds=10,batch_size=batch_size, counter=counter, plot=True, stop_after=35)
    #r['pure_rnnx3_5'] = cv(feats5, target5, groups5, models.pure_rnn_3, name='5',epochs=epochs, folds=10,batch_size=batch_size, counter=counter, plot=True, stop_after=35)
    #r['pure_rrn_do_5'] = cv(feats5, target5, groups5, models.pure_rnn_do, name='5',epochs=epochs, folds=10,batch_size=batch_size, counter=counter, plot=True, stop_after=35)
    #r['ann_rrn_5'] = cv(feats5, target5, groups5, models.ann_rnn, name='5',epochs=epochs, folds=10,batch_size=batch_size, counter=counter, plot=True, stop_after=35)
    #with open('results_recurrent_architectures.pkl', 'wb') as f:
    #            pickle.dump(r, f)
    
    #%%   seqlen
    #r = dict()
    ##r['pure_rrn_do_1'] = cv(feats1, target1, groups1, models.pure_rnn_do, name='1',epochs=epochs, folds=10,batch_size=batch_size, counter=counter, plot=True, stop_after=35)
    #for i in [1,2,3,4,5,6,7,8,9,10,15]:
    #    feats_seq, target_seq, group_seq = tools.to_sequences(feats, target, groups, seqlen = i, tolist=False)
    #    r['pure_rrn_do_' + str(i)] = cv(feats_seq, target_seq, group_seq, models.pure_rnn_do_stateful,
    #                                    name=str(i),epochs=epochs, folds=10, batch_size=batch_size, 
    #                                    counter=counter, plot=True, stop_after=35)
    ##
    #with open('results_recurrent_seqlen1-15.pkl', 'wb') as f:
    #            pickle.dump(r, f)
    #%%
#    %%   seqlen = 6, 5-fold
#    batch_size = 512
#    feats_seq, target_seq, group_seq = tools.to_sequences(feats_all, target, groups=groups, seqlen = 6, tolist=False)
#    r = keras_utils.cv(feats_seq, target_seq, group_seq, models.pure_rnn_do, epochs=250, folds=5, batch_size=batch_size, name='RNN',
#                                    counter=counter, plot=True, stop_after=15, balanced=False)
#    results.update(r)
#    with open('results_recurrent', 'wb') as f:
#                pickle.dump(results, f)
###    
##    
    #%%   seqlen = 6, 5-fold mit past
    #r = dict()
    #batch_size = 1440
    #feats_seq, target_seq, group_seq = tools.to_sequences(feats, target, groups, seqlen = 6, tolist=False)
    #r['pure_rrn_do_6_weighted-'] = cv(feats_seq, target_seq, group_seq, models.pure_rnn_do,
    #                                name=str(6),epochs=300, folds=10, batch_size=batch_size, 
    #                                counter=counter, plot=True, stop_after=15, weighted=True)
    #r['pure_rrn_do_6_not_weighted-'] = cv(feats_seq, target_seq, group_seq, models.pure_rnn_do,
    #                                name=str(6),epochs=300, folds=10, batch_size=batch_size, 
    #                                counter=counter, plot=True, stop_after=15, weighted=False)
    ##
    #with open('results_recurrent_seqlen-6-w5.pkl', 'wb') as f:
    #            pickle.dump(r, f)
    #%%
    #s
    batch_size = 256
    epochs = 250
    name = 'LSTM moreL2'
    ###
    rnn = {'model':models.pure_rnn_do, 'layers': ['fc1'],  'seqlen':6,
           'epochs': 250,  'batch_size': 512,  'stop_after':15, 'balanced':False}
    print(rnn)
    data = data[:12000]
    target = target[:12000]
    groups = groups[:12000]
#    model = 'C:\\Users\\Simon\\dropbox\\Uni\\Masterthesis\\AutoSleepScorer\\weights\\balanced'
    model = models.cnn3adam_filter_morel2
    r = keras_utils.cv (data, target, groups, model, rnn=rnn,trans_tuple=trans_tuple, name=name,
                             epochs=epochs, folds=5, batch_size=batch_size, counter=counter,
                             plot=True, stop_after=15, balanced=False, cropsize=2800)
    results.update(r)
    with open('results_recurrejsuttestingnt_morel2.pkl', 'wb') as f:
                pickle.dump(results, f)
    telegram_send.send(parse_mode='Markdown',messages=['DONE {} {}\n```\n{}\n```\n'.format(os.path.basename(__file__),name, tools.print_string(results))])
            