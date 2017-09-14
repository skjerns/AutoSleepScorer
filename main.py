# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 12:04:50 2017

Main script for the AutoSleepScorer

@author: Simon Kern, sikern[ett]uos.de
"""
import os

if __name__ == '__main__':
    import sleepscorer
import sleeploader
import numpy as np
import argparse


    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Predict sleep stages from sleep eeg')
    parser.add_argument('psg', action='store', help='Path to eeg header or a directory containing several eeg files', type=str)
    parser.add_argument('-ext','--psg_file_extension', action='store', help='File extension for directory usage', default='*', type=str)

    parser.add_argument("-cnn", '--cnn_weights', help="path to cnn weigths",  action="store", default='./weights/cnn.hdf5', type=str)
    parser.add_argument("-rnn", '--rnn_weights', help="path to rnn weigths",  action="store", default='./weights/rnn.hdf5', type=str)

    parser.add_argument("-eeg", help="channel to use as EEG",  action="store", default=False, type=str)
    parser.add_argument("-emg", help="channel to use as EMG",  action="store", default=False, type=str)
    parser.add_argument("-eog", help="channel to use as EOG",  action="store", default=False, type=str)
    parser.add_argument("-refeeg", help="reference to use for EEG",  action="store", default=False, type=str)
    parser.add_argument("-refemg", help="reference to use for EMG",  action="store", default=False, type=str)
    parser.add_argument("-refeog", help="reference to use for EOG",  action="store", default=False, type=str)


    args = parser.parse_args()
    cnn = args.cnn_weights
    rnn = args.rnn_weights
    ext = args.psg_file_extension
    path = args.psg
    eeg = args.eeg
    emg = args.emg
    eog = args.eog
    refeeg = args.refeeg
    refemg = args.refemg
    refeog = args.refeog
    
    # Check arguments
    if os.path.isdir(path):
        files = os.listdir(path)
        if ext == '*':
            files =  [os.path.join(path,s) for s in files if os.path.splitext(s)[1][1:] in ['vhdr', 'edf', 'fif','gz','egi','set','data', 'bdf', 'cnt','sqd', 'ds']]
            print('{} EEG files found'.format(len(files)))
        else:
            files =  [s for s in files if s.endswith(ext)]
            print('{} EEG files found with extension .{}'.format(len(files), ext))

    elif os.path.isfile(path):
        files = [path]
    else:
        raise FileNotFoundError('Path {} is not found or not a directory'.format(path))
        
    if not os.path.isfile(cnn):
        raise FileNotFoundError('CNN weights not found at {}'.format(cnn))
        
    if not os.path.isfile(rnn):
        raise FileNotFoundError('RNN weights not found at {}'.format(rnn))
    
    #    cnn = 'C:/Users/Simon/dropbox/Uni/Masterthesis/AutoSleepScorerDev/weights/1207LSTM moreL2cnn3adam_filter_morel2_4_0.861-0.774'
#    rnn = 'C:/Users/Simon/dropbox/Uni/Masterthesis/AutoSleepScorerDev/weights/1207LSTM moreL2fc1_4_0.895-0.825'
    scorer = sleepscorer.AutoSleepScorer(files, cnn, rnn, hypnograms=True)
    scorer.run()
#
##    
##    
##    
#    true = np.genfromtxt('C:/sleep/cshs100/cshhs_test_run1_100_ccshs-trec-1800191-profusion.xml_eegonset_0_to_10000_min.txt')[:,0]
#    true[true==4]=3
#    true[true==5]=4
#    print(np.mean(true==np.argmax(preds,1)))
#    scorer.predict(sleep[:,:2800,:])
#    preds1 = scorer.get_probas()[:len(true)]
#    scorer.predict(sleep[:,100:2900,:])
#    preds2 = scorer.get_probas()[:len(true)]
#    scorer.predict(sleep[:,200:,:])
#    preds3 = scorer.get_probas()[:len(true)]
#    
#    preds = np.argmax((preds1+preds2+preds3)/3,1)
#    print(np.mean(preds==true))
#
