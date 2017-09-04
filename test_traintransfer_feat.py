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
import time
from copy import deepcopy
import scipy
import keras_utils
import models
import pickle
import telegram_send
import sleeploader
import matplotlib; matplotlib.rcParams['figure.figsize'] = (10, 3)
np.random.seed(42)

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

if __name__ == '__main__':
    import argparse
    import matplotlib; matplotlib.use('Agg')
    import numpy as np
    parser = argparse.ArgumentParser(description='Start calculating that dataset.')
    parser.add_argument('dataset', metavar='Dataset', type=str, help='which data set to run')
    parser.add_argument('norm', metavar='Normalization', type=str, help='which normalization to use')
    args = parser.parse_args()
    trainset = args.dataset
    normmode = args.norm

    
    datadir = './data' if os.name == 'posix' else 'd:\\sleep\\data'
    testsets = [pkl[:-4] for pkl in os.listdir(datadir)]
    
    print('Training on {}, testing on {} with normalization {}'.format(trainset, testsets, normmode))
    plot=False
    data,target,groups = load_data(trainset, norm = 'group' if normmode=='group' else 'all')
    if normmode == 'all': 
        normmode = 'all'
    elif normmode == 'group': 
        normmode = 'group'
    elif normmode=='zmap':
        normmode = trainset
    else:
        raise SyntaxError( 'mode {} not recognized'.format(normmode))
    
    feats = tools.get_all_features(data)
    feats = scipy.stats.zscore(feats)

    ann, lstm = keras_utils.train_models_feat(feats, target, groups)
    ann.save(os.path.join('.','weights','traintransfer_featann_{}_{}'.format(trainset,normmode)), include_optimizer=False)
    lstm.save(os.path.join('.','weights','traintransfer_featlstm_{}_{}'.format(trainset,normmode)), include_optimizer=False)
    results = {}
    for dataset in testsets:
        gc.collect()
        data,target,groups = load_data(dataset, norm=normmode)
        feats = tools.get_all_features(data)
        feats = scipy.stats.zscore(feats)
        print('testing...')
        results[dataset] = keras_utils.test_data_ann_rnn(feats,target,groups,ann,lstm)
        ann_acc, ann_f1, rnn_acc, rnn_f1, confmat, preds = results[dataset]
        tools.plot_results_per_patient(preds[0],preds[1],preds[2], title='Results per Patient for {}'.format(dataset), fname='per_patient_{}_{}'.format(dataset,normmode))
        print('CNN acc: {:.1f} CNN f1: {:.1f} LSTM acc: {:.1f} LSTM f1: {:.1f}'.format(ann_acc*100,ann_f1*100,rnn_acc*100,rnn_f1*100))
    
    with open('results_transfer_feat_{}_{}'.format(trainset, normmode), 'wb') as f:
        pickle.dump(results,f)

    telegram_send.send(parse_mode='Markdown',messages=['DONE {}  {} {}\n```\n{}\n```\n'.format(os.path.basename(__file__), trainset, normmode, tools.print_string(results))])

    
    
    