# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 12:04:50 2017

Main script for the AutoSleepScorer

@author: Simon Kern, sikern[ett]uos.de
"""
import os

import numpy as np
import argparse
from . import sleeploader
from . import tools
if __name__ == '__main__':
    from . import sleepscorer


    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Predict sleep stages from sleep eeg')
    parser.add_argument('psg', action='store', help='Path to eeg header or a directory containing several eeg files', type=str)
    parser.add_argument('-ext','--psg_file_extension', action='store', help='File extension for directory usage', default='*', type=str)

    parser.add_argument("-cnn", '--cnn_weights', help="path to cnn weigths",  action="store", default=None, type=str)
    parser.add_argument("-rnn", '--rnn_weights', help="path to rnn weigths",  action="store", default=None, type=str)

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
        
    if cnn is not None and not os.path.isfile(cnn) :
        raise FileNotFoundError('CNN weights not found at {}'.format(cnn))
        
    if  rnn is not None and not os.path.isfile(rnn):
        raise FileNotFoundError('RNN weights not found at {}'.format(rnn))
    
    #    cnn = 'C:/Users/Simon/dropbox/Uni/Masterthesis/AutoSleepScorerDev/weights/1207LSTM moreL2cnn3adam_filter_morel2_4_0.861-0.774'
#    rnn = 'C:/Users/Simon/dropbox/Uni/Masterthesis/AutoSleepScorerDev/weights/1207LSTM moreL2fc1_4_0.895-0.825'
    scorer = sleepscorer.AutoSleepScorer(files, cnn, rnn, hypnograms=True)
    scorer.run()

    
def run_example():
    if not os.path.isfile('sample-psg.edf'):
        print('A sample file from the EDFx database will be loaded...')
        tools.download('https://physionet.nlm.nih.gov/pn4/sleep-edfx/SC4001E0-PSG.edf', 'sample-psg.edf')
        tools.download('https://pastebin.com/raw/jbzz16wP', 'sample-psg.hypnogram.csv') 
        
    file = 'sample-psg.edf'
    scorer = sleepscorer.Scorer([file], hypnograms=True)
    scorer.run()
    tools.show_sample_hypnogram('sample-psg.hypnogram.csv')  
    
    
    sleepdata = sleeploader.SleepData('sample-psg.edf', start = 2880000, stop = 5400000, channels={'EEG':'EEG Fpz-Cz', 'EMG':'EMG submental', 'EOG':'EOG horizontal'}, preload=False)
    scorer = sleepscorer.Scorer([sleepdata], hypnograms=True)
    scorer.run()
    tools.show_sample_hypnogram('sample-psg.hypnogram.csv', start=960, stop=1800)  
    
#run_example()