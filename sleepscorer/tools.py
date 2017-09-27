# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 13:33:45 2016

@author: Simon

These are tools for the AutoSleepScorer.
"""

import numpy as np
import os.path
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
from scipy import stats
from scipy.signal import butter, lfilter
from sklearn.metrics import f1_score, accuracy_score
from copy import deepcopy
import os,sys
import re

from urllib.request import urlretrieve, urlopen


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


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]      
        
       
def plot_confusion_matrix(fname, conf_mat, target_names, 
                          title='', cmap='Blues', perc=True,figsize=[6,5],cbar=True):
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
    plt.xlabel('Time after recording start')
    plt.ylabel('Sleep Stage')
    plt.title(title)
    plt.tight_layout()

    
def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r{}% {:.1f}/{:.1f} MB".format(int(percent), readsofar/1024**2, totalsize/1024**2)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))

def download(url, file = None):
    if file:
        urlretrieve(url, file, reporthook)
    else:
        return urlopen(url).read().decode("utf-8")

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
   
    
def show_sample_hypnogram(filename, start=None,stop=None, title='True Sleep Stage'):
    hypno = np.loadtxt(filename)
    hypno = hypno[start:stop]
    hypno[hypno==4]=3
    hypno[hypno==5]=4
    plot_hypnogram(hypno,title = title)













