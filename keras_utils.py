# -*- coding: utf-8 -*-
import os, sys
import time
import keras
import tools
import pickle
import models
import numpy as np
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import GroupKFold
from tensorflow.python.client import device_lib
import tensorflow as tf
from tqdm import tqdm
import warnings
from keras.layers.core import Lambda


#%%
print('##################################################')
print('##################################################')
print(__name__)
print('##################################################')
print('##################################################')

def get_available_gpus():
    """
    The function does what its name says. Simple as that.
    """
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])   

global MAX_GPU
MAX_GPU  = get_available_gpus()
print("{} GPUs available".format(MAX_GPU))


def make_parallel(model, gpu_count=-1):
    """
    Distributes a model on two GPUs by creating a copy on each GPU
    and running slices of the batches on each GPU,
    then the two models outputs are merged again.
    Attention: The batches that are fed to the model must always be the same size
    @param (keras.models.Model) model: The model that should be distributed
    @param (int) gpu_count:            number of GPUs to use
    @return model
    """
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)
    if gpu_count == -1: gpu_count = get_available_gpus()
    if gpu_count < 2: return model # If only 1 GPU, nothing needs to be done.
    print('Making model parallel on {} GPUs'.format(gpu_count))
    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            # merged.append(keras.layers.merge(outputs, mode='concat', concat_axis=0))
            merged.append(keras.layers.concatenate(outputs, axis = 0))
        model = keras.models.Model(input=model.inputs, output=merged)
        model.multi_gpu = True
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[keras.metrics.categorical_accuracy])
        return model
    
    

def get_activations(model, data, layer, batch_size=256, flatten=True, cropsize=0, verbose=0):
#    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
#                                      [model.layers[layername].output if type(layername) is type(int) else model.get_layer(layername).output])
#    

    if type(layer) is str:
        layerindex = None
        layername  = layer
    else:
        layerindex = layer
        layername  = None   
#    print (layername, layerindex)
    
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.get_layer(name=layername, index=layerindex).output])

    activations = []
    batch_size = int(batch_size)
    for i in tqdm(range(int(np.ceil(len(data)/batch_size))), desc='Feature extraction') if verbose==1 else range(int(np.ceil(len(data)/batch_size))):
        batch = np.array(data[i*batch_size:(i+1)*batch_size], dtype=np.float32)
        if cropsize!=0:
            diff = (batch.shape[1]-cropsize)//2
            batch = batch[:,diff:-diff,:]
        act = get_layer_output([batch,0])[0]
        activations.extend(act)
    activations = np.array(activations, dtype=np.float32)
    if flatten: activations = activations.reshape([len(activations),-1])
    return activations

def train_models(data, targets, groups,model=None, cropsize=2800, batch_size=512, epochs=250, epochs_to_stop=15,rnn_epochs_to_stop=15):
    """
    trains a cnn3adam_filter_l2 model with a LSTM on top on 
    the given data with 20% validation set and returns the two models
    """
    input_shape = list((np.array(data[0])).shape) #train_data.shape
    input_shape[0] = cropsize
    n_classes = targets.shape[1]
    train_idx, val_idx = GroupKFold(5).split(groups,groups,groups).__next__()
    train_data   = [data[i] for i in train_idx]
    train_target = targets[train_idx]
    train_groups = groups[train_idx]
    val_data     = [data[i] for i in val_idx]
    val_target   = targets[val_idx]
    val_groups   = groups[val_idx]
    model = models.cnn3adam_filter_l2(input_shape, n_classes) if model is None else model(input_shape, n_classes)
    g_train= generator(train_data, train_target, batch_size, val=False, cropsize=cropsize)
    g_val  = generator(val_data, val_target, batch_size, val=True, cropsize=cropsize)
    cb  = Checkpoint_balanced(g_val, verbose=1, groups=val_groups,
                              epochs_to_stop=epochs_to_stop, plot = True, name = '{}, {}'.format(model.name, 'testing'))
    model.fit_generator(g_train, g_train.n_batches, epochs=epochs, callbacks=[cb], max_queue_size=1, verbose=0)
    val_acc = cb.best_acc
    val_f1  = cb.best_f1
    print('CNN Val acc: {:.1f}, Val F1: {:.1f}'.format(val_acc*100, val_f1*100))
    
    # LSTM training
    rnn_modelfun = models.pure_rnn_do
    lname = 'fc1'
    seq = 6
    rnn_epochs = epochs
    stopafter_rnn = rnn_epochs_to_stop
    features = get_activations(model, train_data + val_data, lname, batch_size*2, cropsize=cropsize)
    train_data_extracted = features[0:len(train_data)]
    val_data_extracted   = features[len(train_data):]
    assert (len(train_data)==len(train_data_extracted)) and (len(val_data)==len(val_data_extracted))
    train_data_seq, train_target_seq, train_groups_seq = tools.to_sequences(train_data_extracted, train_target,groups=train_groups, seqlen=seq)
    val_data_seq, val_target_seq, val_groups_seq       = tools.to_sequences(val_data_extracted,   val_target,  groups=val_groups, seqlen=seq)
    rnn_shape  = list((np.array(train_data_seq[0])).shape)
    neurons = int(np.sqrt(rnn_shape[-1])*4)
    rnn_model  = rnn_modelfun(rnn_shape, n_classes, layers=2, neurons=neurons, dropout=0.3)
    print('Starting RNN model with input from layer fc1: {} at {}'.format(rnn_model.name, rnn_shape, time.ctime()))
    g_train= generator(train_data_seq, train_target_seq, batch_size, val=False)
    g_val  = generator(val_data_seq, val_target_seq, batch_size, val=True)
    cb = Checkpoint_balanced(g_val, verbose=1, groups = val_groups_seq, 
                             epochs_to_stop=stopafter_rnn, plot = True, name = '{}, {}'.format(rnn_model.name,'fc1'))         
    rnn_model.fit_generator(g_train, g_train.n_batches, epochs=rnn_epochs, verbose=0, callbacks=[cb],max_queue_size=1)    
    val_acc = cb.best_acc
    val_f1  = cb.best_f1
    print('LSTM Val acc: {:.1f}, Val F1: {:.1f}'.format(val_acc*100, val_f1*100))

    return model, rnn_model
    
    
def train_models_feat(data, targets, groups, batch_size=512, epochs=250, epochs_to_stop=15):
    """
    trains a ann and rnn model with features
    the given data with 20% validation set and returns the two models
    """
    batch_size = 512
    input_shape = list((np.array(data[0])).shape) #train_data.shape
    n_classes = targets.shape[1]
    train_idx, val_idx = GroupKFold(5).split(groups,groups,groups).__next__()
    train_data   = [data[i] for i in train_idx]
    train_target = targets[train_idx]
    train_groups = groups[train_idx]
    val_data     = [data[i] for i in val_idx]
    val_target   = targets[val_idx]
    val_groups   = groups[val_idx]
    model = models.ann(input_shape, n_classes)
    g_train= generator(train_data, train_target, batch_size, val=False)
    g_val  = generator(val_data, val_target, batch_size, val=True)
    cb  = Checkpoint_balanced(g_val, verbose=1, groups=val_groups,
                              epochs_to_stop=epochs_to_stop, plot = True, name = '{}, {}'.format(model.name, 'testing'))
    model.fit_generator(g_train, g_train.n_batches, epochs=epochs, callbacks=[cb], max_queue_size=1, verbose=0)
    val_acc = cb.best_acc
    val_f1  = cb.best_f1
    print('CNN Val acc: {:.1f}, Val F1: {:.1f}'.format(val_acc*100, val_f1*100))
    
    # LSTM training
    batch_size = 512
    n_classes = targets.shape[1]
    train_idx, val_idx = GroupKFold(5).split(groups,groups,groups).__next__()
    train_data   = np.array([data[i] for i in train_idx])
    train_target = targets[train_idx]
    train_groups = groups[train_idx]
    val_data     = np.array([data[i] for i in val_idx])
    val_target   = targets[val_idx]
    val_groups   = groups[val_idx]
    
    train_data_seq, train_target_seq, train_groups_seq = tools.to_sequences(train_data, train_target, groups=train_groups, seqlen=6)
    val_data_seq, val_target_seq, val_groups_seq        = tools.to_sequences(val_data, val_target, groups=val_groups, seqlen=6)
    
    input_shape = list((np.array(train_data_seq[0])).shape) #train_data.shape
    print(input_shape)
    rnn_model = models.pure_rnn_do(input_shape, n_classes)
    
    g_train = generator(train_data_seq, train_target_seq, batch_size, val=False)
    g_val   = generator(val_data_seq, val_target_seq, batch_size, val=True)
    cb  = Checkpoint_balanced(g_val, verbose=1, groups=val_groups_seq,
                              epochs_to_stop=epochs_to_stop, plot = True, name = '{}, {}'.format(rnn_model.name, 'testing'))
    rnn_model.fit_generator(g_train, g_train.n_batches, epochs=epochs, callbacks=[cb], max_queue_size=1, verbose=0)
    val_acc = cb.best_acc
    val_f1  = cb.best_f1
    print('CNN Val acc: {:.1f}, Val F1: {:.1f}'.format(val_acc*100, val_f1*100))
    
    

    return model, rnn_model   
    

def test_data_ann_rnn(feats, target, groups, ann, rnn):
    """
    mode = 'scores' or 'preds'
    take two ready trained models (cnn+rnn)
    test on input data and return acc+f1
    """
    if target.ndim==2: target = np.argmax(target,1)

        

    cnn_pred = ann.predict_classes(feats, 1024, verbose=0)

    cnn_acc = accuracy_score(target, cnn_pred)
    cnn_f1  = f1_score(target, cnn_pred, average='macro')
    
    seqlen = rnn.input_shape[1]
    features_seq, target_seq, groups_seq = tools.to_sequences(feats, target, seqlen=seqlen, groups=groups)
    new_targ_seq = np.roll(target_seq, 4)
    rnn_pred = rnn.predict_classes(features_seq, 1024, verbose=0)
    rnn_acc = accuracy_score(new_targ_seq, rnn_pred)
    rnn_f1  = f1_score(new_targ_seq,rnn_pred, average='macro')
    confmat = confusion_matrix(new_targ_seq, rnn_pred)
    return [cnn_acc, cnn_f1, rnn_acc, rnn_f1, confmat, (rnn_pred, target_seq, groups_seq)]


def test_data_cnn_rnn(data, target, groups, cnn, rnn, layername='fc1', cropsize=2800, verbose=1, only_lstm = False):
    """
    mode = 'scores' or 'preds'
    take two ready trained models (cnn+rnn)
    test on input data and return acc+f1
    """
    if target.ndim==2: target = np.argmax(target,1)
    if cropsize != 0: 
        diff = (data.shape[1] - cropsize)//2
        data = data[:,diff:-diff:,:]
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if only_lstm == False:
            cnn_pred = cnn.predict_classes(data, 1024,verbose=0)
        else:
            cnn_pred = target
        features = get_activations(cnn, data, 'fc1', verbose=verbose)
    
        cnn_acc = accuracy_score(target, cnn_pred)
        cnn_f1  = f1_score(target, cnn_pred, average='macro')
        
        seqlen = rnn.input_shape[1]
        features_seq, target_seq, groups_seq = tools.to_sequences(features, target, seqlen=seqlen, groups=groups)
        new_targ_seq = np.roll(target_seq, 4)
        rnn_pred = rnn.predict_classes(features_seq, 1024, verbose=0)
        rnn_acc = accuracy_score(new_targ_seq, rnn_pred)
        rnn_f1  = f1_score(new_targ_seq,rnn_pred, average='macro')
        confmat = confusion_matrix(new_targ_seq, rnn_pred)
   
    return [cnn_acc, cnn_f1, rnn_acc, rnn_f1, confmat, (rnn_pred, target_seq, groups_seq)]




#%%
class Checkpoint_balanced(keras.callbacks.Callback):
    """
    Callback routine for Keras
    Calculates accuracy and f1-score on the validation data
    Implements early stopping if no improvement on validation data for X epochs
    """
    def __init__(self,val_gen, bal_gen=None, train_gen=None, counter = 0, verbose=0, 
                 groups = False, epochs_to_stop=15, plot = False, name = ''):
        super(Checkpoint_balanced, self).__init__()
        
        assert (bal_gen is None) == (train_gen is None), 'Must give train generator if balanced generator is present'
        self.balanced = False if bal_gen is None else True
        self.bgen = bal_gen
        self.tgen = train_gen
        self.gen = val_gen
        self.groups = groups
        self.nclasses = val_gen.Y.shape[-1]
        self.best_weights = None
        self.verbose = verbose
        self.counter = counter
        self.plot = plot
        self.epochs_to_stop = epochs_to_stop
        self.figures = []
        self.name = name
        self.not_improved=0
        self.best_f1 = 0
        self.best_acc = 0
        self.best_epoch = 0
              
    def on_train_begin(self, logs={}):
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_f1 = []
        self.val_acc = []
        self.per_class = [[] for i in range(self.nclasses)]
        self.start = time.time()
        
        self.not_improved=0
        self.best_f1 = 0
        self.best_acc = 0
        self.best_epoch = 0
        if self.plot: 
            self.figures.append(plt.figure(figsize=(15,8)))
            
    def on_epoch_end(self, epoch, logs={}):
        self.gen.reset() # to be sure
        if self.balanced:
            self.tgen.reset()
            y_tpred = np.array(self.model.predict_generator(self.tgen, self.tgen.n_batches, max_q_size=1))
            self.bgen.pmatrix = y_tpred
            
        y_pred = np.array(self.model.predict_generator(self.gen, self.gen.n_batches, max_q_size=1))
        y_true = self.gen.Y
        
        f1 = f1_score(np.argmax(y_true,1),np.argmax(y_pred,1), average="macro")
        acc = accuracy_score(np.argmax(y_true,1),np.argmax(y_pred,1))
#        val_loss = keras.metrics.categorical_crossentropy(y_true, np.argmax(y_pred))
        val_loss = log_loss(y_true, y_pred)
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('categorical_accuracy'))
        self.val_loss.append(val_loss)
        self.val_f1.append(f1)
        self.val_acc.append(acc)

        if f1 > self.best_f1:
            self.not_improved = 0
            self.best_f1 = f1
            self.best_acc = acc
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            if self.verbose==1: print('+', end='')
        else:
            self.not_improved += 1
            if self.verbose==1: print('.', end='', flush=True)
            if self.not_improved > self.epochs_to_stop and self.epochs_to_stop:
                print("\nNo improvement after epoch {}".format(epoch), flush=True)
                self.model.stop_training = True

        y_pred = np.argmax(y_pred,1)
        y_true = np.argmax(y_true,1)
        for i in range(self.nclasses):
            self.per_class[i].append(np.mean((y_true[y_true==i])==(y_pred[y_true==i])))
        confmat = confusion_matrix(y_true, y_pred)    
            

        if self.plot:
            plt.clf()
            plt.subplot(2,2,1)
            plt.plot(self.loss)
            plt.plot(self.val_loss, 'r')
            plt.title('Loss')
            plt.legend(['loss', 'val_loss'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.subplot(2,2,2)
            plt.plot(self.val_acc)
            plt.plot(self.val_f1)
            plt.legend(['val acc', 'val f1'])
            plt.xlabel('Epoch')
            plt.ylabel('%')
            plt.title('Best: acc {:.1f}, f1 {:.1f}'.format(self.best_acc*100,self.best_f1*100))
            plt.subplot(2,3,4)
            for i in range(self.nclasses):
                plt.plot(self.per_class[i])
            plt.legend(['W', 'S1', 'S2', 'SWS', 'REM'])
            plt.title('Per class accuracy')
            plt.subplot(2,3,5)
            if self.groups is not False:
                tools.plot_results_per_patient(y_pred,y_true, self.groups,fname='')
            plt.subplot(2,3,6)
            tools.plot_confusion_matrix('',confmat,['W', 'S1', 'S2', 'SWS', 'REM'])
            plt.title('Epoch {}'.format(len(self.loss)))
            plt.suptitle(self.name)
            plt.show()
            plt.pause(0.0001)
        if self.verbose == 2:
            print('Epoch {}: , current: {:.1f}/{:.1f}, best: {:.1f}/{:.1f}'.format(epoch, acc*100, f1*100, self.best_acc*100 , self.best_f1*100))
        
    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_weights)
        try: self.model.save('copy.model')
        except Exception: print('could not save model')
        if self.verbose > 0: print(' {:.1f} min'.format((time.time()-self.start)/60), flush=True)
        if self.plot:
            filename ='{}_{}_{}.png'.format(self.counter, self.name, self.model.name)
            filename = ''.join([x if x not in ',;\\/:><|?*\"' else '_' for x in filename])
            try: plt.savefig(os.path.join('.','plots', filename ))
            except Exception as e:print('can\'t save plots: {}'.format(e))
#        try:
#            self.model.save(os.path.join('.','weights', str(self.counter) + self.model.name))
#        except Exception as error:
#            print("Got an error while saving model: {}".format(error))
#        return
    

    

#%%
class generator_balanced(object):
    """
        Data generator util for Keras. 
        Generates data in such a way that it can be used with a stateful RNN

    :param X: data (either train or val) with shape 
    :param Y: labels (either train or val) with shape 
    :param num_of_batches: number of batches (np.ceil(len(Y)/batch_size) for non truncated mode 
    :param sequential: for stateful training
    :param truncate: Only yield full batches, in sequential mode the batches will be wrapped in case of false
    :param val: it only returns the data without the target
    :param random: randomize pos or neg within a batch, ignored in sequential mode 
    
    :return: patches (batch_size, 15, 15, 15) and labels (batch_size,)
    """
    def __init__(self, X, Y, batch_size, cropsize=0):
        
        assert len(X) == len(Y), 'X and Y must be the same length {}!={}'.format(len(X),len(Y))
        print('starting balanced generator')
        self.X = X
        self.Y = Y
        self.cropsize=cropsize
        self.batch_size = int(batch_size)
        self.pmatrix = np.ones_like(self.Y)
        self.reset()
        
    def reset(self):
        """ Resets the state of the generator"""
        self.step = 0
        Y = np.argmax(self.Y,1)
        labels = np.unique(Y)
        idx = []
        smallest = len(Y)
        for i,label in enumerate(labels):
            where = np.where(Y==label)[0]
            if smallest > len(where): 
                self.slabel = i
                smallest = len(where)
            idx.append(where)
        self.idx = idx
        self.labels = labels
        self.n_per_class = int(self.batch_size // len(labels))
        self.n_batches = int(np.ceil((smallest//self.n_per_class)))+1
        self.update_probabilities()
        
    def update_probabilities(self):
        Y = np.argmax(self.Y,1)
        p = []
        for label in self.labels:
            where = np.where(Y==label)[0]
            proba = self.pmatrix[:, label][where]
            proba = 1-(proba / np.sum(proba))
            proba = proba / np.sum(proba)
            p.append(proba)
        self.p = p
    
    def __next__(self):
        if self.step==self.n_batches:
            self.reset()
        x_batch = []
        y_batch = []
        for label in self.labels:
            idx = self.idx[label]
            if len(idx)< self.n_per_class:
                x_batch.extend([self.X[i] for i in idx])
                y_batch.extend([self.Y[i] for i in idx])
                self.idx[label] = []
            else:
                number = self.n_per_class# if label!=self.slabel else self.n_per_class
                indexes      = np.random.choice(np.arange(idx.size), number, replace = False)
                choice = idx[indexes]
                x_batch.extend([self.X[i] for i in choice])
                y_batch.extend([self.Y[i] for i in choice])
                self.idx[label] = np.delete (self.idx[label], indexes)
                self.p[label]   = np.delete (self.p[label], indexes)
                self.p[label]   = self.p[label] / np.sum(self.p[label])
#                if label in [0,2,3,4]:
#                    idx = self.idx[label]
#                    indexes_hard    = np.random.choice(np.arange(idx.size), int(number*0.1), p=self.p[label], replace = False)
#                    choice = idx[indexes_hard]
#                    x_batch.extend([self.X[i] for i in choice])
#                    y_batch.extend([self.Y[i] for i in choice])
#                    self.idx[label] = np.delete (self.idx[label], indexes_hard)
#                    self.p[label]   = np.delete (self.p[label], indexes_hard)
#                    self.p[label]   = self.p[label] / np.sum(self.p[label])
                
        diff = len(x_batch[0]) - self.cropsize
        
        if self.cropsize!=0:
            start = np.random.choice(np.arange(0,diff+5,5), len(x_batch))
            x_batch = [x[start[i]:start[i]+self.cropsize,:] for i,x in enumerate(x_batch)]

        x_batch = np.array(x_batch, dtype=np.float32)
        y_batch = np.array(y_batch, dtype=np.int32)
    
        self.step+=1
        return (x_batch, y_batch)  
        



         
class generator(object):
    """
        Data generator util for Keras. 
        Generates data in such a way that it can be used with a stateful RNN

    :param X: data (either train or val) with shape 
    :param Y: labels (either train or val) with shape 
    :param num_of_batches: number of batches (np.ceil(len(Y)/batch_size) for non truncated mode 
    :param sequential: for stateful training
    :param truncate: Only yield full batches, in sequential mode the batches will be wrapped in case of false
    :param val: it only returns the data without the target
    :param random: randomize pos or neg within a batch, ignored in sequential mode 
    
    :return: patches (batch_size, 15, 15, 15) and labels (batch_size,)
    """
    def __init__(self, X, Y, batch_size,cropsize=0, truncate=False, sequential=False,
                 random=True, val=False, class_weights=None):
        
        assert len(X) == len(Y), 'X and Y must be the same length {}!={}'.format(len(X),len(Y))
        if sequential: print('Using sequential mode')
        print ('starting normal generator')
        self.X = X
        self.Y = Y
        self.rnd_idx = np.arange(len(Y))
        self.Y_last_epoch = []
        self.val = val
        self.step = 0
        self.i = 0
        self.cropsize=cropsize
        self.truncate = truncate
        self.random = False if sequential or val else random
        self.batch_size = int(batch_size)
        self.sequential = sequential
        self.c_weights = class_weights if class_weights else dict(zip(np.unique(np.argmax(Y,1)),np.ones(len(np.argmax(Y,1)))))
        assert set(np.argmax(Y,1)) == set([int(x) for x in self.c_weights.keys()]), 'not all labels in class weights'
        self.n_batches = int(len(X)//batch_size if truncate else np.ceil(len(X)/batch_size))
        if self.random: self.randomize()
            
        
    def reset(self):
        """ Resets the state of the generator"""
        self.step = 0
        self.Y_last_epoch = []
        
        
    def randomize(self):
        self.X, self.Y, self.rnd_idx = shuffle(self.X, self.Y, self.rnd_idx)
        
    def get_Y(self):
        """ Get the last targets that were created/shuffled"""
        print('This feature is disabled. Please access generator.Y without shuffling.')
        if self.sequential or self.truncate:
            y_len = self.n_batches * self.batch_size
        else:
            y_len = len(self.Y)
        if self.val and (len(self.Y)!=y_len): print('not same length!')
        return np.array(self.Y_last_epoch, dtype=np.int32)[:y_len]
#        return np.array([x[0] for x in self.Y_last_epoch])
    
    def __next__(self):
        self.i +=1
        if self.step==self.n_batches:
            self.step = 0
            if self.random: self.randomize()
        if self.sequential: 
            return self.next_sequential()
        else:
            return self.next_normal()
        
    def next_normal(self):
        x_batch = self.X[self.step*self.batch_size:(self.step+1)*self.batch_size]
        y_batch = self.Y[self.step*self.batch_size:(self.step+1)*self.batch_size]
        
        diff = len(x_batch[0]) - self.cropsize
        if self.cropsize!=0 and not self.val:
            start = np.random.choice(np.arange(0,diff+5,5), len(x_batch))
            x_batch = [x[start[i]:start[i]+self.cropsize,:] for i,x in enumerate(x_batch)]
        elif self.cropsize !=0 and self.val:
            x_batch = [x[diff//2:diff//2+self.cropsize] for i,x in enumerate(x_batch)]
            
        x_batch = np.array(x_batch, dtype=np.float32)
        y_batch = np.array(y_batch, dtype=np.int32)
        self.step+=1
        if self.val:
            self.Y_last_epoch.extend(y_batch)
            return x_batch # for validation generator, save the new y_labels
        else:
            weights = np.ones(len(y_batch))
            for t in np.unique(np.argmax(y_batch,1)):
                weights[np.argmax(y_batch,1)==t] = self.c_weights[t]
            return (x_batch,y_batch)  
        
    def next_sequential(self):
        x_batch = np.array([self.X[(seq * self.n_batches + self.step) % len(self.X)] for seq in range(self.batch_size)], dtype=np.float32)
        y_batch = np.array([self.Y[(seq * self.n_batches + self.step) % len(self.X)] for seq in range(self.batch_size)], dtype=np.int32)
        if self.cropsize !=0:
            x_batch = [x[diff:diff+self.cropsize] for i,x in enumerate(x_batch)]
            y_batch = [x[diff:diff+self.cropsize] for i,x in enumerate(y_batch)]
        
        self.step+=1
        if self.val:
            self.Y_last_epoch.extend(y_batch)
            return x_batch # for validation generator, save the new y_labels
        else:
            return (x_batch,y_batch) 
        
#%%
def cv(data, targets, groups, modfun, rnn=False, trans_tuple=None, epochs=250, folds=5, batch_size=256,
       val_batch_size=0, stop_after=0, name='', counter=0, plot = False, balanced=False, cropsize=0):
    """
    Crossvalidation routinge for an RNN using extracted features on a basemodel
    :param rnns: list with the following:
                 [rnnfun, [layernames], seqlen, batch_size]
    :param ...: should be self-explanatory

    :returns results: dictionary with all RNN results
    """
    if val_batch_size == 0: val_batch_size = batch_size
    input_shape = list((np.array(data[0])).shape) #train_data.shape
    if cropsize!=0:  input_shape[0] = cropsize
    n_classes = targets.shape[1]
    
    if type(modfun) == str:
        wpath = modfun
        modfun = False

    gcv = GroupKFold(folds)
    dict_id    = modfun.__name__ + name if modfun else 'cnn' + '_' + name
    results    = {dict_id:[]}
    if rnn:
        for lname in rnn['layers']:
            results[name + '_' + lname] = []
    
    for i, idxs in enumerate(gcv.split(groups, groups, groups)):
        K.clear_session()
        print('-----------------------------')
        print('Starting fold {}: {}-{} at {}'.format(i+1,modfun.__name__ if modfun else 'cnn', name, time.ctime()))
        train_idx, test_idx = idxs
        sub_cv = GroupKFold(folds)
        train_sub_idx, val_idx = sub_cv.split(groups[train_idx], groups[train_idx], groups[train_idx]).__next__()
        val_idx      = train_idx[val_idx]  
        train_idx    = train_idx[train_sub_idx]
        
        train_data   = [data[i] for i in train_idx]
        train_target = targets[train_idx]
        train_groups = groups[train_idx]
        val_data     = [data[i] for i in val_idx]
        val_target   = targets[val_idx]
        val_groups   = groups[val_idx]
        test_data    = [data[i] for i in test_idx]       
        test_target  = targets[test_idx]
        test_groups  = groups[test_idx]
        
        
        if modfun:
            model = modfun(input_shape, n_classes)
        else:
            fold = os.listdir(wpath)[i]
            model = keras.models.load_model(os.path.join(wpath,fold))
            
            
        modelname = model.name
        lname = modelname
        g_train= generator(train_data, train_target, batch_size*2, val=True, cropsize=cropsize)
        g_val  = generator(val_data, val_target, batch_size*2, val=True, cropsize=cropsize)
        g_test = generator(test_data, test_target, batch_size*2, val=True, cropsize=cropsize)
        
        if balanced:
            g = generator_balanced(train_data, train_target, batch_size,cropsize=cropsize)
            cb  = Checkpoint_balanced(g_val, g, g_train, verbose=1, counter=counter, groups=val_groups,
                 epochs_to_stop=stop_after, plot = plot, name = '{}, {}, fold: {}'.format(name,lname,i))
        else:
            g = generator(train_data, train_target, batch_size, random=True, cropsize=cropsize)
            cb  = Checkpoint_balanced(g_val, verbose=1, counter=counter, groups=val_groups,
                 epochs_to_stop=stop_after, plot = plot, name = '{}, {}, fold: {}'.format(name,lname,i))
        
        if modfun: model.fit_generator(g, g.n_batches, epochs=epochs, callbacks=[cb], max_queue_size=1, verbose=0)
        

        
        y_pred = model.predict_generator(g_test, g_test.n_batches, max_queue_size=1)
        y_true = g_test.Y
        val_acc = cb.best_acc
        val_f1  = cb.best_f1
        test_acc = accuracy_score(np.argmax(y_true,1),np.argmax(y_pred,1))
        test_f1  = f1_score(np.argmax(y_true,1),np.argmax(y_pred,1), average="macro")
        confmat = confusion_matrix(np.argmax(y_true,1),np.argmax(y_pred,1))
        if plot:
            plt.subplot(2,3,5)
            plt.cla()
            tools.plot_results_per_patient(y_pred,y_true, test_groups, fname='')
            plt.title('Test Cases')
            plt.subplot(2,3,6)
            plt.cla()
            tools.plot_confusion_matrix('',confmat,['W', 'S1', 'S2', 'SWS', 'REM'], cbar=False)
            plt.title('Test conf. Acc: {:.1f} F1: {:.1f}'.format(test_acc*100, test_f1*100))
            plt.show()
            plt.pause(0.0001)
        results[dict_id].append([cb.best_acc, cb.best_f1, test_acc, test_f1, confmat])
        
        if modfun: # only save if we calculated the results
            try: model.save(os.path.join('.','weights', str(counter) + name + model.name + '_' + str(i) + "_{:.3f}-{:.3f}".format(test_acc,test_f1)))
            except Exception as error:  print("Got an error while saving model: {}".format(error))
        print('ANN results: val acc/f1: {:.5f}/{:.5f}, test acc/f1: {:.5f}/{:.5f}'.format(cb.best_acc, cb.best_f1, test_acc, test_f1))
        ##########
        if trans_tuple is not None:
            trans_data,trans_target,trans_groups = trans_tuple
            g_trans = generator(trans_data, trans_target, batch_size*2, val=True, cropsize=cropsize)
            y_trans = model.predict_generator(g_trans, g_trans.n_batches, max_queue_size=1)
            t_trans = g_trans.Y
            trans_acc = accuracy_score(np.argmax(t_trans,1),np.argmax(y_trans,1))
            trans_f1  = f1_score(np.argmax(t_trans,1),np.argmax(y_trans,1), average="macro")
            print('Transfer ANN results: acc/f1: {:.5f}/{:.5f}'.format( trans_acc, trans_f1))
        ##########
        if rnn:
            rnn_modelfun = rnn['model'] 
            layernames = rnn['layers']
            seq = rnn['seqlen']
            rnn_bs = rnn['batch_size']
            rnn_epochs = rnn['epochs']
            stopafter_rnn = rnn['stop_after']
            
            for lname in layernames:
                extracted = get_activations(model, train_data + val_data + test_data, lname, batch_size*2, cropsize=cropsize)
                train_data_extracted = extracted[0:len(train_data)]
                val_data_extracted   = extracted[len(train_data):len(train_data)+len(val_data)]
                test_data_extracted  = extracted[len(train_data)+len(val_data):]
                assert (len(train_data)==len(train_data_extracted)) and (len(test_data)==len(test_data_extracted)) and (len(val_data)==len(val_data_extracted))
                train_data_seq, train_target_seq, train_groups_seq = tools.to_sequences(train_data_extracted, train_target,groups=train_groups, seqlen=seq)
                val_data_seq, val_target_seq, val_groups_seq       = tools.to_sequences(val_data_extracted,   val_target,  groups=val_groups, seqlen=seq)
                test_data_seq, test_target_seq, test_groups_seq    = tools.to_sequences(test_data_extracted,  test_target, groups=test_groups, seqlen=seq)
             
                rnn_shape  = list((np.array(train_data_seq[0])).shape)
                neurons = 100
                print('Starting RNN model with input from layer {}: {} at {}'.format(lname, rnn_shape, time.ctime()))
                rnn_model  = rnn_modelfun(rnn_shape, n_classes, layers=2, neurons=neurons, dropout=0.3)
                
                g_val  = generator(val_data_seq, val_target_seq, rnn_bs*2, val=True)
                g_test = generator(test_data_seq, test_target_seq, rnn_bs*2, val=True)
                g_train= generator(train_data_seq, train_target_seq, batch_size*2, val=True)
                if rnn['balanced']:
                    g = generator_balanced(train_data_seq, train_target_seq, rnn_bs)
                    cb = Checkpoint_balanced(g_val, g, g_train, verbose=1, counter=counter, groups = val_groups_seq, 
                         epochs_to_stop=stopafter_rnn, plot = plot, name = '{}, {}, fold: {}'.format(name,lname,i))
                else:              
                    g = generator(train_data_seq, train_target_seq, rnn_bs)
                    cb = Checkpoint_balanced(g_val, verbose=1, counter=counter, groups = val_groups_seq, 
                         epochs_to_stop=stopafter_rnn, plot = plot, name = '{}, {}, fold: {}'.format(name,lname,i))
                
                rnn_model.fit_generator(g, g.n_batches, epochs=rnn_epochs, verbose=0, callbacks=[cb],max_queue_size=1)    
                y_pred = rnn_model.predict_generator(g_test, g_test.n_batches, max_queue_size=1)
                y_true = g_test.Y
                val_acc = cb.best_acc
                val_f1  = cb.best_f1
                test_acc = accuracy_score(np.argmax(y_true,1),np.argmax(y_pred,1))
                test_f1  = f1_score(np.argmax(y_true,1),np.argmax(y_pred,1), average="macro")
                confmat = confusion_matrix(np.argmax(y_true,1),np.argmax(y_pred,1))
                try: rnn_model.save(os.path.join('.','weights', str(counter)+ name + lname + '_' + str(i) + "_{:.3f}-{:.3f}".format(test_acc,test_f1)))
                except Exception as error:  print("Got an error while saving model: {}".format(error))
                if plot:
                        plt.subplot(2,3,5)
                        plt.cla()
                        tools.plot_results_per_patient(y_pred,y_true, test_groups_seq,fname='')
                        plt.title('Test Cases')
                        plt.subplot(2,3,6)
                        plt.cla()
                        tools.plot_confusion_matrix('',confmat,['W', 'S1', 'S2', 'SWS', 'REM'], cbar=False)
                        plt.title('Test conf. Acc: {:.1f} F1: {:.1f}'.format(test_acc*100, test_f1*100))
                        plt.show()
                        plt.pause(0.0001)
                results[name + '_' + lname].append([val_acc, val_f1, test_acc, test_f1, confmat])
                print('fold {}: val acc/f1: {:.5f}/{:.5f}, test acc/f1: {:.5f}/{:.5f}'.format(i,cb.best_acc, cb.best_f1, test_acc, test_f1))
                ##########
                if trans_tuple is not None:
                    trans_data,trans_target,trans_groups = trans_tuple
                    extracted = get_activations(model, trans_data, lname, batch_size*2, cropsize=cropsize)
                    trans_data, trans_target,trans_groups = tools.to_sequences(extracted, trans_target, groups=trans_groups, seqlen=seq)
                    g_trans = generator(trans_data, trans_target, batch_size*2, val=True, cropsize=0)
                    y_trans = rnn_model.predict_generator(g_trans, g_trans.n_batches, max_queue_size=1)
                    t_trans = g_trans.Y
                    trans_acc = accuracy_score(np.argmax(t_trans,1),np.argmax(y_trans,1))
                    trans_f1  = f1_score(np.argmax(t_trans,1),np.argmax(y_trans,1), average="macro")
                    print('Transfer LSTM results: acc/f1: {:.5f}/{:.5f}'.format( trans_acc, trans_f1))
                ##########
            save_dict = {'1 Number':counter,
                         '2 Time':time.ctime(),
                         '3 CV':'{}/{}.'.format(i+1, folds),
                         '5 Model': lname,
                         '100 Comment': name,
                         '10 Epochs': epochs,
                         '11 Val acc': '{:.2f}'.format(val_acc*100),
                         '12 Val f1': '{:.2f}'.format(val_f1*100),
                         '13 Test acc':'{:.2f}'.format( test_acc*100),
                         '14 Test f1': '{:.2f}'.format(test_f1*100),
                         'Test Conf': str(confmat).replace('\n','')}
            tools.save_results(save_dict=save_dict)
        
        
        try:
            with open('{}_{}_results.pkl'.format(counter, dict_id), 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            print("Error while saving results: ", e)
        sys.stdout.flush()
        
    return results

#%%
def test_model(data, targets, groups, testdata, modfun, epochs=250, batch_size=512,
       val_batch_size=0, stop_after=0, name='', counter=0, plot = True):
    """
    Train a model on the data
    :param ...: should be self-explanatory
    :returns results: (best_val_acc, best_val_f1, best_test_acc, best_test_f1)
    """
    
    return
#%%

print('loaded keras_utils.py')

