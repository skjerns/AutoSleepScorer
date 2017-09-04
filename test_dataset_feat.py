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
import scipy
import models
import pickle
import keras_utils
import sleeploader
import matplotlib; matplotlib.rcParams['figure.figsize'] = (10, 3)
np.random.seed(42)

if __name__ == '__main__':
    import argparse
    import matplotlib;matplotlib.use('Agg')
    import numpy as np
    parser = argparse.ArgumentParser(description='Start calculating that dataset.')
    parser.add_argument('modus', metavar='Dataset', type=str, help='which data set to run')
    args = parser.parse_args()
    dataset = args.modus 
    print(dataset)
    plot = False if os.name == 'posix' else True
    try:
        time.sleep(np.random.rand()) # prevent access problems of parallel scripts
        with open('count') as f:
            counter = int(f.read())
    except IOError:
        print('No previous experiment?')
        counter = 0
    with open('count', 'w') as f:
      f.write(str(counter+1))        
    
    #%%

    def load_data(dataset):
        datadir = './data' if os.name == 'posix' else 'd:\\sleep\\data'
        sleep = sleeploader.SleepDataset(datadir)
        sleep.load_object(str(dataset))
        data, target, groups = sleep.get_all_data(groups=True)
        data = tools.normalize(data)
        target[target==4] = 3
        target[target==5] = 4
        target[target==6] = 0
        idx = np.where(target==8)[0]
        data = np.delete(data, idx, axis=0)
        groups = np.delete(groups, idx)
        target = np.delete(target, idx)
        target = keras.utils.to_categorical(target)
        return data, target, groups
    
    data,target,groups = load_data(dataset)

    #%%
    #s
    batch_size = 256
    epochs = 250
    name =  dataset
    model = models.pure_rnn_do
    data = tools.get_all_features(data)
    feats = scipy.stats.zscore(data)
    feats_seq, target_seq, groups_seq = tools.to_sequences(data,target,groups=groups, seqlen=6)
    results = keras_utils.cv (feats_seq, target_seq, groups_seq, model, name=name,
                             epochs=epochs, folds=5, batch_size=batch_size, counter=counter,
                             plot=plot, stop_after=15, balanced=False)

    with open('results_dataset_feat_{}'.format(dataset), 'wb') as f:
                pickle.dump(results, f)
