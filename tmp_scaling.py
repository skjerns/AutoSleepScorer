# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:32:43 2017

@author: Simon
"""

if not '__file__' in vars(): __file__= u'C:/Users/Simon/dropbox/Uni/Masterthesis/AutoSleepScorer/main.py'
import os
import gc; gc.collect()
import matplotlib
matplotlib.use('Agg')
import numpy as np
import keras
import tools
import time
from copy import deepcopy
import keras_utils
import models
import pickle
import scipy
from tqdm import tqdm
import telegram_send
import sleeploader
from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt import fmin, tpe
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, log_loss
import matplotlib; matplotlib.rcParams['figure.figsize'] = (10, 3)
np.random.seed(42)
import warnings

def load_data(dataset, norm = 'none'):
    datadir = './data' if os.name == 'posix' else 'd:\\sleep\\data'
    sleep = sleeploader.SleepDataset(datadir)
    sleep.load_object(str(dataset))
    data, target, groups = sleep.get_all_data(groups=True)
    target[target==4] = 3
    target[target==5] = 4
    target[target>5]  = 0
    if norm is 'all':
        print('Normalizing {} over whole set\n...'.format(dataset), end='')
        data = tools.normalize(data)
    elif norm == 'none':
        pass
    elif norm == 'group':
        print('Normalizing {} per patient\n...'.format(dataset), end='')
        data = tools.normalize(data, groups=groups)
    else:
        print('Normalizing {} with reference {}\n...'.format(dataset, norm), end='')
        compdata,_,_  = load_data(norm, norm='none') 
        print('...', end='')
        data = tools.normalize(data, comp=compdata)
        
    target = keras.utils.to_categorical(target)
    return deepcopy(data), deepcopy(target), deepcopy(groups)

def get_f1(scale):
    global best_f1
#    idx = np.random.choice(np.arange(len(crop)), 10000 if len(target)>10000 else len(target), replace=False)
    idx = np.arange(len(target))
#    pred = cnn.predict_proba((crop[idx])/scale, 1024, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = keras_utils.test_data_cnn_rnn((crop[idx])/scale, target, groups, cnn, rnn, verbose=0, only_lstm = True, cropsize=0)
        f1 = res[3]
        acc= res[2]
#        f1_score(np.argmax(target[idx],1), np.argmax(pred,1), average='macro')
    print(res[2],f1)
    return -acc

if __name__ == '__main__':
    cnn = keras.models.load_model(os.path.join('.','weights', 'traintransfer_cnn_cshs50_all'))
    rnn = keras.models.load_model(os.path.join('.','weights', 'traintransfer_lstm_cshs50_all'))
    cnn.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001))
    rnn.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001))
    results = {}
    datasets = ['vinc', 'emsaad', 'emsach', 'edfx', 'cshs100']
    for dataset in datasets:
        print('### starting {} ###'.format(dataset))
        data, target, groups = load_data(dataset, 'none')
        crop = data[:,100:2900,:]
        best_f1 = 0
#        space = (hp.normal('scale', np.std(data), 10)) 
#        best = fmin(get_f1, space, algo=tpe.suggest, max_evals=500)
        space = (hp.normal('scale', np.std(data), 10)) 
        best = scipy.optimize.fmin(get_f1, x0 = np.std(data), maxfun = 100, retall=True, ftol=0.005, xtol=0.01)
        print('---results---')
        best_scale = best[0]
        results[dataset] = keras_utils.test_data_cnn_rnn((data)/best_scale, target, groups, cnn, rnn, verbose=0)
        results[dataset].append('scale:{}'.format(best_scale))
        print('CNN {:.2f}/{:.2f}, LSTM {:.2f}/{:.2f}'.format(results[dataset][0]*100,results[dataset][1]*100,results[dataset][2]*100,results[dataset][3]*100))
   
    with open('results_scaling.pkl', 'wb') as f:
        pickle.dump(results,f)
    
    telegram_send.send(messages=['DONE {}'.format(os.path.basename(__file__))])
    telegram_send.send(parse_mode='Markdown',messages=['```\n{}\n```'.format(tools.print_string(results))])

    