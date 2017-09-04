# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 13:33:45 2016

@author: Simon

These are tools for the AutoSleepScorer.
"""

import csv
import numpy as np
import os.path
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import prettytable
import time
from multiprocessing import Pool, cpu_count
from scipy import fft
from scipy import stats
from scipy.signal import butter, lfilter
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score
from copy import deepcopy
import json
import os
import re

use_sfreq=100.0



def plot_signal(data1,data2):
    class Visualizer():
        def __init__(self, data1,data2):
            self.data1 = data1
            self.data2 = data2
            self.pos = 0
            self.scale = 0

             
        def update(self):
            try:
#                self.ax.subplot(121)
                self.ax.cla()
                self.ax.plot(self.data1[self.pos,:,0])
                self.ax.plot(self.data2[self.pos,:,0])
                self.ax.set_ylim([-30,30])
                plt.title(self.pos)
                self.fig.canvas.draw()
                
            except Exception as e: print(e)
    def key_event(e):
        v = _vis
        try:
            if e.key == "right":
                v.pos +=1
            elif e.key == "left":
                v.pos -= 1
            elif e.key == "up":
                v.scale += 1
            elif e.key == "down":
                v.scale -= 1
            else:
                print(e.key)

            v.update()
            
        except Exception as e:print(e)
    global _vis
    _vis = Visualizer(data1,data2)
    _vis.fig = plt.figure()
    _vis.fig.canvas.mpl_connect('key_press_event', key_event)
    _vis.ax = _vis.fig.add_subplot(111)
    _vis.update()

def confmat_to_numpy(confmat_str):
    rows = confmat_str.split('] [')
    new_array = []
    for s in rows:
        s = s.replace('[[','')
        s = s.replace(']]','')
        s = s.split(' ')
        s = [int(x) for x in s if x is not '']
        new_array.append(s)
    return np.array(new_array)
        
    
def convert_Y_to_seq_batches(Y, batch_size):
    if (len(Y)%batch_size)!= 0: Y = Y[:-(len(Y)%batch_size)]
    idx = np.arange(len(Y))
    idx = idx.reshape([batch_size,-1]).flatten('F')
    return Y[idx]

def test(data, *args):
    if args is not (): assert np.all([len(data)==len(x) for x in args])


def to_sequences(data, *args, groups=False, seqlen = 0, tolist = True, wrap = False):
    '''
    Creates time-sequences
    
    :param groups: Only creates sequences with members of the same group
    :returns list: list of list of numpy arrays. this way no memory redundance is created
    '''
    if seqlen==0: return data if args is  () else [data]+list(args)
    if seqlen==1: return  np.expand_dims(data,1) if args is () else [np.expand_dims(data,1)]+list(args)
    if args is not (): assert np.all([len(data)==len(x) for x in args]), 'Data and Targets must have same length'
    if groups is not False: assert len(data)==len(groups), 'Data and Groups must have the same size {}:{}'.format(len(data), len(groups))
    assert data.shape[0] > seqlen, 'Future steps must be smaller than number of datapoints'
    
    data = [x for x in data] # transform to lists
    new_data = []
    for i in range((len(data))if wrap else (len(data)-seqlen+1) ):
        seq = []
        for s in range(seqlen):
            seq.append(data[(i+s)%len(data)])
        new_data.append(seq)
        
    if groups is not False:
        new_groups = []
        for i in range((len(groups))if wrap else (len(groups)-seqlen+1) ):
            seq = []
            for s in range(seqlen):
                seq.append(groups[(i+s)%len(groups)])
            new_groups.append(seq)
        new_groups = np.array(new_groups)
        overlap = (np.min(new_groups,1)!=np.max(new_groups,1))
        remove_idx = set(np.where(overlap)[0])
        new_groups = np.array([v for i, v in enumerate(new_groups) if i not in remove_idx])[:,0]
        new_data   = [v for i, v in enumerate(new_data) if i not in remove_idx]
        
    if not tolist: 
        new_data = np.array(new_data, dtype=np.float32)  
        
    if args is not ():
        new_data = [new_data]
        for array in args:
            new_array = np.roll(array, -seqlen+1)
            if not wrap: new_array = new_array[:-seqlen+1]
            if groups is not False:  new_array = [v for i, v in enumerate(new_array) if i not in remove_idx]
            assert len(new_array)==len(new_data[0]), 'something went wrong {}!={}'.format(len(new_array),len(new_data[0])) 
            new_data.append(np.array(new_array))
        if groups is not False:
            new_data.append(new_groups)
        
    
    return new_data




def label_to_one_hot(y):
    '''
    Convert labels into "one-hot" representation
    '''
    n_values = np.max(y) + 1
    y_one_hot = np.eye(n_values)[y]
    return y_one_hot




def normalize(signals, axis=None, groups=None, MP = False, comp=None):
    """
    :param signals: 1D, 2D or 3D signals
    returns zscored per patient
    """
    
    if comp is not None:
        print('zmapping with axis {}'.format(axis))
        return stats.zmap(signals, comp , axis=axis)
    
    if groups is None: 
        print('zscoring with axis {}'.format(axis))
        return stats.zscore(signals, axis=axis)
    
    if signals.ndim == 1: signals = np.expand_dims(signals,0) 
    if signals.ndim == 2: signals = np.expand_dims(signals,2)
    
    if MP:
        print('zscoring per patient using {} cores'.format(cpu_count()))
        p = Pool(cpu_count()) #use all except for one
        res = []
        new_signals = np.zeros_like(signals)
        for ID in np.unique(groups):
            idx = groups==ID
            job = p.apply_async(stats.zscore, args = (signals[idx],), kwds={'axis':None})
            res.append(job)
        start = 0
        for r in res:
            values = r.get(timeout = 1200)
            end = start + len(values)
            new_signals[start:end] = values
            end = start
        return new_signals
    else:
        print('zscoring per patient')
        res = []
        for ID in np.unique(groups):
            idx = groups==ID
            job = stats.zscore(signals[idx], axis=None)
            res.append(job)
        new_signals = np.vstack(res)
        return new_signals



def future(signals, fsteps):
    """
    adds fsteps points of the future to the signal
    :param signals: 2D or 3D signals
    :param fsteps: how many future steps should be added to each data point
    """
    if fsteps==0: return signals
    assert signals.shape[0] > fsteps, 'Future steps must be smaller than number of datapoints'
    if signals.ndim == 2: signals = np.expand_dims(signals,2) 
    nsamp = signals.shape[1]
    new_signals = np.zeros((signals.shape[0],signals.shape[1]*(fsteps+1), signals.shape[2]),dtype=np.float32)
    for i in np.arange(fsteps+1):
        new_signals[:,i*nsamp:(i+1)*nsamp,:] = np.roll(signals[:,:,:],-i,axis=0)
    return new_signals.squeeze() if new_signals.shape[2]==1 else new_signals



def feat_eeg(signals):
    """
    calculate the relative power as defined by Leangkvist (2012),
    assuming signal is recorded with 100hz
    """
    if signals.ndim == 1: signals = np.expand_dims(signals,0)
    
    sfreq = use_sfreq
    nsamp = float(signals.shape[1])
    feats = np.zeros((signals.shape[0],9),dtype='float32')
    # 5 FEATURE for freq babnds
    w = (fft(signals,axis=1)).real
    delta = np.sum(np.abs(w[:,np.arange(0.5*nsamp/sfreq,4*nsamp/sfreq, dtype=int)]),axis=1)
    theta = np.sum(np.abs(w[:,np.arange(4*nsamp/sfreq,8*nsamp/sfreq, dtype=int)]),axis=1)
    alpha = np.sum(np.abs(w[:,np.arange(8*nsamp/sfreq,13*nsamp/sfreq, dtype=int)]),axis=1)
    beta  = np.sum(np.abs(w[:,np.arange(13*nsamp/sfreq,20*nsamp/sfreq, dtype=int)]),axis=1)
    gamma = np.sum(np.abs(w[:,np.arange(20*nsamp/sfreq,50*nsamp/sfreq, dtype=int)]),axis=1)   # only until 50, because hz=100
    spindle = np.sum(np.abs(w[:,np.arange(12*nsamp/sfreq,14*nsamp/sfreq, dtype=int)]),axis=1)
    sum_abs_pow = delta + theta + alpha + beta + gamma + spindle
    feats[:,0] = delta /sum_abs_pow
    feats[:,1] = theta /sum_abs_pow
    feats[:,2] = alpha /sum_abs_pow
    feats[:,3] = beta  /sum_abs_pow
    feats[:,4] = gamma /sum_abs_pow
    feats[:,5] = spindle /sum_abs_pow
    feats[:,6] = np.log10(stats.kurtosis(signals, fisher=False, axis=1))        # kurtosis
    feats[:,7] = np.log10(-np.sum([(x/nsamp)*(np.log(x/nsamp)) for x in np.apply_along_axis(lambda x: np.histogram(x, bins=8)[0], 1, signals)],axis=1))  # entropy.. yay, one line...
    #feats[:,7] = np.polynomial.polynomial.polyfit(np.log(f[np.arange(0.5*nsamp/sfreq,50*nsamp/sfreq, dtype=int)]), np.log(w[0,np.arange(0.5*nsamp/sfreq,50*nsamp/sfreq, dtype=int)]),1)
    feats[:,8] = np.dot(np.array([3.5,4,5,7,30]),feats[:,0:5].T ) / (sfreq/2-0.5)
    if np.any(feats==np.nan): print('NaN detected')
    return np.nan_to_num(feats)



def feat_wavelet(signals):
    """
    calculate the relative power as defined by Leangkvist (2012),
    assuming signal is recorded with 100hz
    """
    if signals.ndim == 1: signals = np.expand_dims(signals,0)
    
    sfreq = use_sfreq
    nsamp = float(signals.shape[1])
    feats = np.zeros((signals.shape[0],8),dtype='float32')
    # 5 FEATURE for freq babnds
    w = (fft(signals,axis=1)).real
    delta = np.sum(np.abs(w[:,np.arange(0.5*nsamp/sfreq,4*nsamp/sfreq, dtype=int)]),axis=1)
    theta = np.sum(np.abs(w[:,np.arange(4*nsamp/sfreq,8*nsamp/sfreq, dtype=int)]),axis=1)
    alpha = np.sum(np.abs(w[:,np.arange(8*nsamp/sfreq,13*nsamp/sfreq, dtype=int)]),axis=1)
    beta  = np.sum(np.abs(w[:,np.arange(13*nsamp/sfreq,20*nsamp/sfreq, dtype=int)]),axis=1)
    gamma = np.sum(np.abs(w[:,np.arange(20*nsamp/sfreq,50*nsamp/sfreq, dtype=int)]),axis=1)   # only until 50, because hz=100
    sum_abs_pow = delta + theta + alpha + beta + gamma
    feats[:,0] = delta /sum_abs_pow
    feats[:,1] = theta /sum_abs_pow
    feats[:,2] = alpha /sum_abs_pow
    feats[:,3] = beta  /sum_abs_pow
    feats[:,4] = gamma /sum_abs_pow
    feats[:,5] = np.log10(stats.kurtosis(signals,fisher=False,axis=1))        # kurtosis
    feats[:,6] = np.log10(-np.sum([(x/nsamp)*(np.log(x/nsamp)) for x in np.apply_along_axis(lambda x: np.histogram(x, bins=8)[0], 1, signals)],axis=1))  # entropy.. yay, one line...
    #feats[:,7] = np.polynomial.polynomial.polyfit(np.log(f[np.arange(0.5*nsamp/sfreq,50*nsamp/sfreq, dtype=int)]), np.log(w[0,np.arange(0.5*nsamp/sfreq,50*nsamp/sfreq, dtype=int)]),1)
    feats[:,7] = np.dot(np.array([3.5,4,5,7,30]),feats[:,0:5].T ) / (sfreq/2-0.5)
    if np.any(feats==np.nan): print('NaN detected')

    return np.nan_to_num(feats)


def feat_eog(signals):
    """
    calculate the EOG features
    :param signals: 1D or 2D signals
    """

    if signals.ndim == 1: signals = np.expand_dims(signals,0)
    sfreq = use_sfreq
    nsamp = float(signals.shape[1])
    w = (fft(signals,axis=1)).real   
    feats = np.zeros((signals.shape[0],15),dtype='float32')
    delta = np.sum(np.abs(w[:,np.arange(0.5*nsamp/sfreq,4*nsamp/sfreq, dtype=int)]),axis=1)
    theta = np.sum(np.abs(w[:,np.arange(4*nsamp/sfreq,8*nsamp/sfreq, dtype=int)]),axis=1)
    alpha = np.sum(np.abs(w[:,np.arange(8*nsamp/sfreq,13*nsamp/sfreq, dtype=int)]),axis=1)
    beta  = np.sum(np.abs(w[:,np.arange(13*nsamp/sfreq,20*nsamp/sfreq, dtype=int)]),axis=1)
    gamma = np.sum(np.abs(w[:,np.arange(20*nsamp/sfreq,50*nsamp/sfreq, dtype=int)]),axis=1)   # only until 50, because hz=100
    sum_abs_pow = delta + theta + alpha + beta + gamma
    feats[:,0] = delta /sum_abs_pow
    feats[:,1] = theta /sum_abs_pow
    feats[:,2] = alpha /sum_abs_pow
    feats[:,3] = beta  /sum_abs_pow
    feats[:,4] = gamma /sum_abs_pow
    feats[:,5] = np.dot(np.array([3.5,4,5,7,30]),feats[:,0:5].T ) / (sfreq/2-0.5) #smean
    feats[:,6] = np.sqrt(np.max(signals, axis=1))    #PAV
    feats[:,7] = np.sqrt(np.abs(np.min(signals, axis=1)))   #VAV   
    feats[:,8] = np.argmax(signals, axis=1)/nsamp #PAP
    feats[:,9] = np.argmin(signals, axis=1)/nsamp #VAP
    feats[:,10] = np.sqrt(np.sum(np.abs(signals), axis=1)/ np.mean(np.sum(np.abs(signals), axis=1))) # AUC
    feats[:,11] = np.sum(((np.roll(np.sign(signals), 1,axis=1) - np.sign(signals)) != 0).astype(int),axis=1)/nsamp #TVC
    feats[:,12] = np.log10(np.std(signals, axis=1)) #STD/VAR
    feats[:,13] = np.log10(stats.kurtosis(signals,fisher=False,axis=1))       # kurtosis
    feats[:,14] = np.log10(-np.sum([(x/nsamp)*((np.log((x+np.spacing(1))/nsamp))) for x in np.apply_along_axis(lambda x: np.histogram(x, bins=8)[0], 1, signals)],axis=1))  # entropy.. yay, one line...
    if np.any(feats==np.nan): print('NaN detected')
    return np.nan_to_num(feats)


def feat_emg(signals):
    """
    calculate the EMG median as defined by Leangkvist (2012),
    """
    if signals.ndim == 1: signals = np.expand_dims(signals,0)
    sfreq = use_sfreq
    nsamp = float(signals.shape[1])
    w = (fft(signals,axis=1)).real   
    feats = np.zeros((signals.shape[0],13),dtype='float32')
    delta = np.sum(np.abs(w[:,np.arange(0.5*nsamp/sfreq,4*nsamp/sfreq, dtype=int)]),axis=1)
    theta = np.sum(np.abs(w[:,np.arange(4*nsamp/sfreq,8*nsamp/sfreq, dtype=int)]),axis=1)
    alpha = np.sum(np.abs(w[:,np.arange(8*nsamp/sfreq,13*nsamp/sfreq, dtype=int)]),axis=1)
    beta  = np.sum(np.abs(w[:,np.arange(13*nsamp/sfreq,20*nsamp/sfreq, dtype=int)]),axis=1)
    gamma = np.sum(np.abs(w[:,np.arange(20*nsamp/sfreq,50*nsamp/sfreq, dtype=int)]),axis=1)   # only until 50, because hz=100
    sum_abs_pow = delta + theta + alpha + beta + gamma
    feats[:,0] = delta /sum_abs_pow
    feats[:,1] = theta /sum_abs_pow
    feats[:,2] = alpha /sum_abs_pow
    feats[:,3] = beta  /sum_abs_pow
    feats[:,4] = gamma /sum_abs_pow
    feats[:,5] = np.dot(np.array([3.5,4,5,7,30]),feats[:,0:5].T ) / (sfreq/2-0.5) #smean
    emg = np.sum(np.abs(w[:,np.arange(12.5*nsamp/sfreq,32*nsamp/sfreq, dtype=int)]),axis=1)
    feats[:,6] = emg / np.sum(np.abs(w[:,np.arange(8*nsamp/sfreq,32*nsamp/sfreq, dtype=int)]),axis=1)  # ratio of high freq to total motor
    feats[:,7] = np.median(np.abs(w[:,np.arange(8*nsamp/sfreq,32*nsamp/sfreq, dtype=int)]),axis=1)    # median freq
    feats[:,8] = np.mean(np.abs(w[:,np.arange(8*nsamp/sfreq,32*nsamp/sfreq, dtype=int)]),axis=1)   #  mean freq
    feats[:,9] = np.std(signals, axis=1)    #  std 
    feats[:,10] = np.mean(signals,axis=1)
    feats[:,11] = np.log10(stats.kurtosis(signals,fisher=False,axis=1) )
    feats[:,12] = np.log10(-np.sum([(x/nsamp)*((np.log((x+np.spacing(1))/nsamp))) for x in np.apply_along_axis(lambda x: np.histogram(x, bins=8)[0], 1, signals)],axis=1))  # entropy.. yay, one line...
    if np.any(feats==np.nan): print('NaN detected')

    return np.nan_to_num(feats)


def feat_emgmedianfreq(signals):
    """
    calculate the EMG median as defined by Leangkvist (2012),
    """
    if signals.ndim == 1: signals = np.expand_dims(signals,0)
    return np.median(abs(signals),axis=1)


def get_all_features(data):
    """
    returns a vector with extraced features
    :param data: datapoints x samples x dimensions (dimensions: EEG,EMG, EOG)
    """
    eeg = feat_eeg(data[:,:,0])
    emg = feat_emg(data[:,:,1])
    eog = feat_eog(data[:,:,2])
    return np.hstack([eeg,emg,eog])


def get_all_features_m(data):
    """
    returns a vector with extraced features
    :param data: datapoints x samples x dimensions (dimensions: EEG,EMG, EOG)
    """
    p = Pool(3)
    t1 = p.apply_async(feat_eeg,(data[:,:,0],))
    t2 = p.apply_async(feat_eog,(data[:,:,1],))
    t3 = p.apply_async(feat_emg,(data[:,:,2],))
    eeg = t1.get(timeout = 1200)
    eog = t2.get(timeout = 1200)
    emg = t3.get(timeout = 1200)
    p.close()
    p.join()

    return np.hstack([eeg,emg,eog])


def save_results(save_dict=None, **kwargs):
    np.set_printoptions(precision=2,threshold=np.nan)
    if save_dict==None:
        save_dict=kwargs
    for key in save_dict.keys():
        save_dict[key] = str(save_dict[key])
    np.set_printoptions(precision=2,threshold=1000)
    append_json('experiments.json', save_dict)
    jsondict2csv('experiments.json', 'experiments.csv')
    
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]      
        
def jsondict2csv(json_file, csv_file):
    
    key_set = set()
    dict_list = list()
    try:
        with open(json_file,'r') as f:
            for line in f:
                dic = json.loads(line)
                key_set.update(dic.keys())
                dict_list.append(dic)
        keys = list(sorted(list(key_set), key = natural_key))
    
        with open(csv_file, 'w') as f:
            w = csv.DictWriter(f, keys, delimiter=';', lineterminator='\n')
            w.writeheader()
            w.writerows(dict_list)
    except IOError:
        print('could not convert to csv-file. ')
        
    
def append_json(json_filename, dic):
    with open(json_filename, 'a') as f:
        json.dump(dic, f)
        f.write('\n')    


def plot_confusion_matrix(fname, conf_mat, target_names, 
                          title='', cmap='Blues', perc=True,figsize=(6,5),cbar=True):
    """Plot Confusion Matrix."""
    figsize = deepcopy(figsize)
    if cbar == False:
        figsize[0] = figsize[0] - 0.6
    c_names = []
    r_names = []
    if len(target_names) != len(conf_mat):
        target_names = [str(i) for  i in np.arange(len(conf_mat))]
    for i, label in enumerate(target_names):
        c_names.append(label + '\n(' + str(int(np.sum(conf_mat[:,i]))) + ')')
        align = len(str(int(np.sum(conf_mat[i,:])))) + 3 - len(label)
        r_names.append('{:{align}}'.format(label, align=align) + '\n(' + str(int(np.sum(conf_mat[i,:]))) + ')')
        
    cm = conf_mat
    cm = 100* cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    df = pd.DataFrame(data=np.sqrt(cm), columns=c_names, index=r_names)
    if fname != '':plt.figure(figsize=figsize)
    g  = sns.heatmap(df, annot = cm if perc else conf_mat , fmt=".1f" if perc else ".0f",
                     linewidths=.5, vmin=0, vmax=np.sqrt(100), cmap=cmap, cbar=cbar,annot_kws={"size": 13})    
    g.set_title(title)
    if cbar:
        cbar = g.collections[0].colorbar
        cbar.set_ticks(np.sqrt(np.arange(0,100,20)))
        cbar.set_ticklabels(np.arange(0,100,20))
    g.set_ylabel('True sleep stage',fontdict={'fontsize' : 12, 'fontweight':'bold'})
    g.set_xlabel('Predicted sleep stage',fontdict={'fontsize' : 12, 'fontweight':'bold'})
#    plt.tight_layout()
    if fname!='':
        plt.tight_layout()
        g.figure.savefig(os.path.join('plots', fname))


def plot_difference_matrix(fname, confmat1, confmat2, target_names, 
                          title='', cmap='Blues', perc=True,figsize=[5,4],cbar=True,
                          **kwargs):
    """Plot Confusion Matrix."""
    figsize = deepcopy(figsize)
    if cbar == False:
        figsize[0] = figsize[0] - 0.6
    
    cm1 = confmat1
    cm2 = confmat2
    cm1 = 100 * cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
    cm2 = 100 * cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
    cm = cm2 - cm1
    cm_eye = np.zeros_like(cm)
    cm_eye[np.eye(len(cm_eye), dtype=bool)] = cm.diagonal()
    df = pd.DataFrame(data=cm_eye, columns=target_names, index=target_names)
    plt.figure(figsize=figsize)
    g  = sns.heatmap(df, annot=cm, fmt=".1f" ,
                     linewidths=.5, vmin=-10, vmax=10, 
                     cmap='coolwarm_r',annot_kws={"size": 13},cbar=cbar,**kwargs)#sns.diverging_palette(20, 220, as_cmap=True))    
    g.set_title(title)
    g.set_ylabel('True sleep stage',fontdict={'fontsize' : 12, 'fontweight':'bold'})
    g.set_xlabel('Predicted sleep stage',fontdict={'fontsize' : 12, 'fontweight':'bold'})
    plt.tight_layout()

    g.figure.savefig(os.path.join('plots', fname))


def plot_results_per_patient(predictions, targets, groups, title='Results per Patient', fname='results_pp.png'):
    assert len(predictions) ==  len(targets), '{} predictions, {} targets'.format(len(predictions), len(targets))
    IDs = np.unique(groups)
    f1s = []
    accs = []
    if predictions.ndim == 2: predictions = np.argmax(predictions,1)
    if targets.ndim == 2: targets = np.argmax(targets,1)
    statechanges = []
    for ID in IDs:
        y_true = targets [groups==ID]
        y_pred = predictions[groups==ID]
        f1  = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        f1s.append(f1)
        accs.append(acc)
        statechanges.append(np.sum(0!=y_true-np.roll(y_true,1))-1)
    if fname != '':plt.figure()

    plt.plot(f1s,'go')
    plt.plot(accs,'bo')
    if np.min(f1s) > 0.5:
        plt.ylim([0.5,1])
    plt.legend(['F1', 'Acc'])
    plt.xlabel('Patient')
    plt.ylabel('Score')
    if fname is not '':
        title = title + '\nMean Acc: {:.1f} mean F1: {:.1f}'.format(accuracy_score(targets, predictions)*100,f1_score(targets,predictions, average='macro')*100)
    plt.title(title)
#    plt.tight_layout()
    if fname!='':
        plt.savefig(os.path.join('plots', fname))
    return (accs,f1s, statechanges)



def plot_hypnogram(stages, labels=None, title='', ax1=None, **kwargs):
    if labels is None:
        if np.max(stages)==4:
            print('assuming 0=W, 1=S1, 2=S2, 3=SWS, 4=REM')
            labels = ['W', 'S1', 'S2', 'SWS', 'REM']
        if np.max(stages)==5:
            print('assuming 0=W, 1=S1, 2=S2, 3=S3, 4=S4, 5=SWS')
            labels = ['W', 'S1', 'S2', 'S3', 'S4', 'REM']
        if np.max(stages)==8:
            print('assuming 0=W, 1=S1, 2=S2, 3=S3, 4=S4, 5=SWS')
            labels = ['W', 'S1', 'S2', 'S3', 'S4', 'REM', 'Movement']
    labels_dict = dict(zip(np.arange(len(labels)),labels))

    x = []
    y = []
    for i in np.arange(len(stages)):
        s = stages[i]
        if labels_dict[s]=='W':   p = -0
        if labels_dict[s]=='REM': p = -1
        if labels_dict[s]=='S1':  p = -2
        if labels_dict[s]=='S2':  p = -3
        if labels_dict[s]=='SWS': p = -4
        if labels_dict[s]=='S3': p = -4
        if labels_dict[s]=='S4': p = -5
        if i!=0:
            y.append(p)
            x.append(i-1)   
        y.append(p)
        x.append(i)
        
    x = np.array(x)*30
    y = np.array(y)
    if ax1 is None:
        fig = plt.figure(figsize=[8,2])
        ax1 = fig.add_subplot(111)
    formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%H:%M', time.gmtime(s)))
    ax1.xaxis.set_major_formatter(formatter)
    ax1.plot(x,y, **kwargs)
    plt.yticks([0,-1,-2,-3,-4,-5], ['W','REM', 'S1', 'S2', 'SWS' ])
    plt.xticks(np.arange(0,x[-1],3600))
    plt.xlabel('Time after lights-off')
    plt.ylabel('Sleep Stage')
    plt.title(title)
    plt.tight_layout()



def memory():
    from wmi import WMI
    w = WMI('.')
    result = w.query("SELECT WorkingSet FROM Win32_PerfRawData_PerfProc_Process WHERE IDProcess=%d" % os.getpid())
    return int(result[0].WorkingSet)/1024**2
    


def one_hot(hypno, n_categories):
    enc = OneHotEncoder(n_values=n_categories)
    hypno = enc.fit_transform(hypno).toarray()
    return np.array(hypno,'int32')
    
    
def shuffle_lists(*args,**options):
     """ function which shuffles two lists and keeps their elements aligned
         for now use sklearn, maybe later get rid of dependency
     """
     return shuffle(*args,**options)
    

def epoch_voting(Y, chunk_size):
    
    
    Y_new = Y.copy()
    
    for i in range(1+len(Y_new)/chunk_size):
        epoch = Y_new[i*chunk_size:(i+1)*chunk_size]
        if len(epoch) != 0: winner = np.bincount(epoch).argmax()
        Y_new[i*chunk_size:(i+1)*chunk_size] = winner              
    return Y_new

        
def butter_bandpass(lowcut, highpass, fs, order=4):
       nyq = 0.5 * fs
#       low = lowcut / nyq
       high = highpass / nyq
       b, a = butter(order, high, btype='highpass')
       return b, a
   
def butter_bandpass_filter(data, highpass, fs, order=4):
       b, a = butter_bandpass(0, highpass, fs, order=order)
       y = lfilter(b, a, data)
       return y
   
    
def get_freqs (signals, nbins=0):
    """ extracts relative fft frequencies and bins them in n bins
    :param signals: 1D or 2D signals
    :param nbins:  number of bins used as output (default: maximum possible)
    """
    if signals.ndim == 1: signals = np.expand_dims(signals,0)
    sfreq = use_sfreq
    if nbins == 0: nbins = int(sfreq/2)
    
    nsamp = float(signals.shape[1])
    assert nsamp/2 >= nbins, 'more bins than fft results' 
    
    feats = np.zeros((int(signals.shape[0]),nbins),dtype='float32')
    w = (fft(signals,axis=1)).real
    for i in np.arange(nbins):
        feats[:,i] =  np.sum(np.abs(w[:,np.arange(i*nsamp/sfreq,(i+1)*nsamp/sfreq, dtype=int)]),axis=1)
    sum_abs_pow = np.sum(feats,axis=1)
    feats = (feats.T / sum_abs_pow).T
    return feats


def print_string(results_dict):
    """ 
    creates an easy printable string from a results dict
    """
    max_hlen = 42
    hlen  =  7 + len('  '.join(list(results_dict)))
    maxlen = (max_hlen-7) //  len(results_dict) -2
    table = prettytable.PrettyTable(header=True, vrules=prettytable.NONE)
    table.border = False
    table.padding_width = 1
    cv = True if  type(results_dict[list(results_dict.keys())[0]][0]) is list else False
    if cv:
        table.add_column('', ['VAcc', 'V-F1', 'TAcc', 'T-F1'])
    else:
        table.add_column('', ['CAcc', 'C-F1', 'RAcc', 'R-F1'])

    for exp in results_dict:
        res = results_dict[exp]
        scores = []
        if cv:
            for fold in res:
                scores.append(fold[:4])
            scores = np.mean(scores,0)
        else:
            scores = np.array([res[0],res[1],res[2],res[3]])
        table.add_column(exp[0:maxlen] if hlen > max_hlen else exp,['{:.1f}%'.format(x*100) for x in scores])
    return table.get_string()

















