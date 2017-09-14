# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 12:06:04 2017

@author: Simon
"""
import os
import keras
import numpy as np
import tools
import keras.backend as K
import sleeploader
import matplotlib.pyplot as plt


class AutoSleepScorer(object):
    
    def __init__(self, files, cnn, rnn, channels=None, references=None, mapping=None, hypnograms=True):
        self.files = files
        self.q = []
        self.cnn = cnn
        self.rnn = rnn
        self.channels = channels
        self.references = references
        self.hypnograms = hypnograms
    
    def _print(self, string):
        print(string)
    
    
    def add(self, eeg_file):
        if os.path.isfile(eeg_file):
            self.files.append(eeg_file)
        else:
            print('WARNING: Did not add, {} does not exist'.format(eeg_file))
            
    def run(self):
        
        scorer = Scorer()
        self._print('Loading CNN model')
        scorer.load_cnn_model(self.cnn)
        self._print('Loading RNN model')
        scorer.load_rnn_model(self.rnn)
        
        self.q = [x for x in self.files]
        self._print('Starting predictions of {} file(s)'.format(len(self.q)))
        while len(self.q) != 0:
            file = self.q.pop(0)
            self._print('Loading {}'.format(os.path.basename(file)))
            sleep = sleeploader.SleepData(file)()
            self._print('Predicting...')
            preds = scorer.predict(sleep, classes=True)
            np.savetxt(file + '.csv', preds, fmt='%d')
            if self.hypnograms: 
                tools.plot_hypnogram(preds, title = os.path.basename(file))
                plt.savefig(file + '.hyp.png')

            
            
            

            
class Scorer(object):

    def __init__(self, cnn_weights=False, rnn_weights=False, 
                wake = 0, s1 = 1, s2 = 2, sws = 3, rem = 4):
        """
        :param directory: a directory string
        """
        self.predictions = None
        self.mapping = {0:wake, 1:s1, 2:s2, 3:sws, 4:rem}
        
#        if ( cnn_weights and not os.path.isdir(cnn_weights)): raise FileNotFoundError( 'CNN weights {} not found'.format(cnn_weights))
#        if (not rnn_weights and not os.path.isdir(cnn_weights)): raise FileNotFoundError( 'CNN weights {} not found'.format(cnn_weights))
#        if cnn_weights: self.load_cnn_model(cnn_weights)
        
    def _get_activations(self, data, model, layers, batch_size=256):

        layer_list = []
        for layer in layers:
            if type(layer) is str:
                layer_list.append(model.get_layer(layer))
            elif type(layer) is int:
                layer_list.append(model.get_layer(index=layer))
            else:
                raise TypeError('Layername {} has type {}, must be str or int'.format(layer, type(layer)))
        inp = model.input                                           # input placeholder
        outputs = [layer.output for layer in layer_list]          # all layer outputs
        functor = K.function([inp] + [K.learning_phase()], outputs ) # evaluation function
        
        activations = [[] for x in layers]
        for i in range(int(np.ceil(len(data)/batch_size))):
            batch = np.array(data[i*batch_size:(i+1)*batch_size], dtype=np.float32)
            acts = functor([batch,0])
            for i in range(len(acts)):
                activations[i].append(acts[i])
        activations = [np.vstack(x).squeeze() for x in activations]
        
        return activations       


    def load_cnn_model(self, cnn_weights):
        self.cnn = keras.models.load_model(cnn_weights)
    
    
    def load_rnn_model(self, rnn_weights):
        self.rnn = keras.models.load_model(rnn_weights)
        
        
    def predict_cnn(self, data, layers=['fc1', -1], cropsize = 0, batch_size=256, modelpath = None):
        if modelpath is not None: self.load_cnn_model(modelpath)
        cropsize = self.cnn.input_shape[1]
        activations = self._get_activations(data[:,:cropsize,:], self.cnn, layers, batch_size=batch_size)
        if len(layers) == 2:
            features, predictions = activations
            self.features = features
            self.cnn_preds = predictions
            self.predict_rnn(features)
            return features, predictions
        else:
            self.cnn_preds = predictions
            return activations
    
    
    def predict_rnn(self, features, modelpath = None, batch_size=256):
        if modelpath is not None: self.load_rnn_model(modelpath)
        feat_seq = tools.to_sequences(features, seqlen = self.rnn.input_shape[1], tolist=False)
        preds = self.rnn.predict(feat_seq, batch_size=batch_size)
        return preds


    def predict(self, data, classes = True):
        if type(data) is sleeploader.SleepData: data=data()
        feats, cnn_preds = self.predict_cnn(data)
        rnn_preds = self.predict_rnn(feats)
        rnn_preds = np.roll(rnn_preds,-4, axis=0)
        preds = np.vstack([cnn_preds[:len(data)-len(rnn_preds)], rnn_preds]) # fill up missing RNN predictions with CNN preds
        print('Still rolling, I should fix that.')
        if classes:
            preds = np.argmax(preds,1)
            preds = [self.mapping[x] for x in preds]
        return preds        
        
    def get_classes(self):
        
        return np.argmax(self.predictions, 1)
    
    
    def get_probas(self):
        return self.predictions
    
    
    
    
    
    