# -*- coding: utf-8 -*-
import os
import re
import csv
import mne
import pickle as cPickle
import numpy as np
import numpy.random as random
from tools import shuffle, butter_bandpass_filter
from multiprocessing import Pool
from tqdm import trange
from copy import deepcopy

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
         

class SleepDataset(object):



    def __init__(self, directory):
        """
        :param directory: a directory string
        """
        if not os.path.isdir(directory): raise FileNotFoundError( 'Director {} not found'.format(directory))
        self.resample = False
        self.available_channels = []
        self.data = list()
        self.hypno = list()  
        self.directory = directory
        self.loaded = False
        self.shuffle_index = list()
        self.subjects = list()
        self.tqdmloop = False
        self.printed_channels = False
        self.channels   = {'EEG': False,
                           'EMG': False,
                           'EOG': False
                          }
        self.references = {'RefEEG': False,
                           'RefEMG': False,
                           'RefEOG': False
                           }
    
    
    def check_for_normalization(self, data_header):
    
        channels = [c.upper() for c in data_header.ch_names]
        if not data_header.info['sfreq'] == 100 and not self.resample:
            print('WARNING: Data not with 100hz. Use resample=True for resampling')      
        
    
#        if not data_header.info['lowpass'] == 50:
#            print('WARNING: lowpass not at 50')
        if not self.channels['EEG'] in channels:
            print('WARNING: EEG channel missing')    
        if not self.channels['EMG'] in channels:
            print('WARNING: EMG channel missing')
        if not self.channels['EOG'] in channels:
            print('WARNING: EOG channel missing')


        if self.references['RefEEG'] and not self.references['RefEEG'] in channels:
            print('WARNING: EEG channel missing')
        if self.references['RefEMG'] and not self.references['RefEMG'] in channels:
            print('WARNING: EMG channel missing')
        if self.references['RefEOG'] and not self.references['RefEOG'] in channels:
            print('WARNING: EOG channel missing')


        
        
    def load_hypnogram(self, filename, dataformat = '', csv_delimiter='\t', mode='standard'):
        """
        returns an array with sleep stages
        :param filename: loads the given hypno file
        :param mode: standard: just read first row, overwrite = if second row!=0,
                     take that value, concatenate = add values together
        """
        
        h = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9,
             'W':0, 'S1':1, 'S2':2, 'S3':3, 'S4':4, 'SWS':3, 'REM':5, 'A':6, 'M':8}
        
        dataformats = dict({
                            'txt' :'csv',
                            'csv' :'csv',                                           
                            })
        if dataformat == '' :      # try to guess format by extension 
            ext = os.path.splitext(filename)[1][1:].strip().lower()                
            dataformat = dataformats[ext]
            
        if dataformat == 'csv':
            with open(filename) as csvfile:
                reader = csv.reader(csvfile, delimiter=csv_delimiter)
                lhypno = []
                for row in reader:
                    if mode == 'standard':
                        lhypno.append(h[row[0]])
                        
                    elif mode == 'overwrite':
                        if int(row[1]) == 0:
                            lhypno.append(h[row[0]])
                        else:
                            lhypno.append(8)
                            #lhypno.append(int(row[1]))
                            
                    elif mode == 'concatecate':
                        lhypno.append(int(x) for x in row)
        else:
            print('unkown hypnogram format. please use CSV with rows as epoch')        

        lhypno = np.array(lhypno, dtype=np.int32).reshape(-1, 1)
        return lhypno   
        
    
    def load_eeg_header(self,filename, dataformat = '', **kwargs):            # CHECK include kwargs
        dataformats = dict({
                            #'bin' :'artemis123',
                            '???' :'bti',                                           # CHECK
                            'cnt' :'cnt',
                            'ds'  :'ctf',
                            'edf' :'edf',
                            'bdf' :'edf',
                            'sqd' :'kit',
                            'data':'nicolet',
                            'set' :'eeglab',
                            'vhdr':'brainvision',
                            'egi' :'egi',
                            'fif':'fif',
                            'gz':'fif',
                            })
        kwargs['stim_channel'] = None
        if dataformat == '' :      # try to guess format by extension 
            ext = os.path.splitext(filename)[1][1:].strip().lower()  
            dataformat = dataformats[ext]
            
        if dataformat == 'artemis123':
            data = mne.io.read_raw_artemis123(filename, **kwargs)             # CHECK if now in stable release
        elif dataformat == 'bti':
            data = mne.io.read_raw_bti(filename, **kwargs)
        elif dataformat == 'cnt':
            data = mne.io.read_raw_cnt(filename, **kwargs)
        elif dataformat == 'ctf':
            data = mne.io.read_raw_ctf(filename, **kwargs)
        elif dataformat == 'edf':
            data = mne.io.read_raw_edf(filename, **kwargs)
        elif dataformat == 'kit':
            data = mne.io.read_raw_kit(filename, **kwargs)
        elif dataformat == 'nicolet':
            data = mne.io.read_raw_nicolet(filename, **kwargs)
        elif dataformat == 'eeglab':
            data = mne.io.read_raw_eeglab(filename, **kwargs)
        elif dataformat == 'brainvision':                                            # CHECK NoOptionError: No option 'markerfile' in section: 'Common Infos' 
            data = mne.io.read_raw_brainvision(filename, verbose=0)
        elif dataformat == 'egi':
            data = mne.io.read_raw_egi(filename, **kwargs)
        elif dataformat == 'fif':
            data = mne.io.read_raw_fif(filename, **kwargs)
        else: 
            print(['Failed extension not recognized for file: ', filename])           # CHECK throw error here    
          
        if not 'verbose' in  kwargs: print('loaded header ' + filename);
        
        return data
    
    def infer_channels(self, channels, ch_type = 'all'):
        """
        Tries to automatically infer channel names. Very limited functionality.
        
        :param channels: a list of channel names
        :param ch_type: The type of channel that you want to infer (EEG, EMG, EOG or all)
        :returns: tuple(channel, reference) if one channel, dictionary with mappings if all channels
        """
        if not self.printed_channels :
            self._print('Available channels: ' + str(channels))
            self.printed_channels = True
            
        channels = [c.upper() for c in channels]
        
        def infer_eeg(channels):
            # Infer EEG
            ch = False
            ref = False
            if 'EEG' in channels:
                ch = 'EEG'
            elif 'C3' in channels and 'A2' in channels:
                ch = 'C3'
                ref = 'A2'
            elif 'C4' in channels and 'A1' in channels:
                ch = 'C4'
                ref = 'A1'   
            elif 'FPZ' in channels and 'CZ' in channels:
                ch = 'FPZ'
                ref = 'CZ'
            elif 'PZ' in channels and 'OZ' in channels:
                ch = 'PZ'
                ref = 'OZ'
            else:
                for c in channels:
                    if 'C4' in c and 'A1' in c:  
                        ch = c; break
                    if 'C3' in c and 'A2' in c:  
                        ch = c; break
                    if 'EEG' in c: 
                        ch = c; break
            self._print('Infering EEG Channel... {}, Ref: {}'.format(ch, ref))
            return ch, ref
        
        def infer_emg(channels):
            ch = False
            ref = False
            if 'EMG' in channels:
                ch = 'EMG'
                ref = False
            elif 'EMG1' in channels and 'EMG2' in channels:
                ch = 'EMG1'
                ref = 'EMG2'
            else:
                for c in channels:
                    if 'EMG' in c: 
                        ch = c
                        break
            self._print('Infering EMG Channel... {}, Ref: {}'.format(ch, ref))
            return ch, ref
        
        def infer_eog(channels):
            ch = False
            ref = False
            if 'EOG' in channels:
                ch = 'EOG'
            elif 'LOC' in channels:
                ch = 'LOC'
            elif 'ROC' in channels:
                ch = 'ROC'
            elif 'EOG horizontal' in channels:
                ch = 'EOG HORIZONTAL'
            else:
                for c in channels:
                    if 'EOG' in c or 'EYE' in c: 
                        ch = c
                        break
            self._print('Infering EOG Channel... {}, Ref: {}'.format(ch, ref))
            return ch, ref
        

        
        if ch_type.upper() == 'EEG':   return infer_eeg(channels)
        if ch_type.upper() == 'EMG':   return infer_emg(channels)
        if ch_type.upper() == 'EOG':   return infer_eog(channels)
        if ch_type.lower() == 'all':
            eeg, refeeg = infer_eeg(channels)
            emg, refemg = infer_emg(channels)
            eog, refeog = infer_eog(channels)
            return ({'EEG':eeg, 'EMG':emg, 'EOG':eog}, 
                   {'RefEEG': refeeg, 'RefEMG': refemg, 'RefEOG':refeog})
        raise Exception('Infer_channels: Wrong channel type selected: {}'.format(ch_type))
    
    
    def check_channels(self, header):
        channels = [c.upper() for c in header.ch_names]
        filename = os.path.basename(header.filenames[0])
        labels = []
        picks = []
        for ch in self.channels:
            if self.channels[ch].upper() not in channels:
                raise ValueError('ERROR: Channel {} for {} not found in {}\navailable channels: {}'.format(self.channels[ch], ch, filename, channels))
            else:
                picks.append(channels.index(self.channels[ch]))
                labels.append(ch)
        for ch in self.references:
            if not self.references[ch]:continue
            if self.references[ch] not in channels:
                raise ValueError('ERROR: Channel {} for {} not found in {}\navailable channels: {}'.format(self.references[ch], ch, filename, channels))
            else:
                picks.append(channels.index(self.references[ch]))
                labels.append(ch)
        return (picks, labels)
    
    
    def load_hypnopickle(self, filename, path = None):
        """
        loads hypnograms from a pickle file
        """
        if path == None: path = self.directory
        with open(os.path.join(path, filename), 'rb') as f:
            self.hypno, self.hypno_files = cPickle.load(f, encoding='latin1')
            self.subjects = zip(self.eeg_files,self.hypno_files)
            if len(self.hypno) != len(self.data): 
                print('WARNING: {} EEG files and {} Hypno files'.format(len(self.eeg_files),len(self.hypno)))
            else:
                for i in np.arange(len(self.data)):
                    if len(self.data[i])/ self.samples_per_epoch != len(self.hypno[i]):
                        print('WARNING, subject {} has EEG len {} and Hypno len {}'.format(i, len(self.data[i])/ self.samples_per_epoch,len(self.hypno[i])))               
            print ('Loaded hypnogram with {} subjects'.format(len(self.hypno)))
        
        
    def save_hypnopickle(self, filename, path = None):
        """
        saves the current hypnograms to a pickle file
        """
        if path == None: path = self.directory
        with open(os.path.join(path, filename), 'wb') as f:
            cPickle.dump((self.hypno,self.hypno_files),f,2)
        
    
    def load_object(self, filename = 'sleepdata.pkl', path = None):
        """
        saves the entire state of the SleepData object
        """
        
        if not filename [-4:] == '.pkl': filename = filename + '.pkl'
        if path == None: path = self.directory
        try:
            print('Loading data from {}'.format(filename))
            with open(os.path.join(path, filename), 'rb') as f:
                tmp_dict = cPickle.load(f, fix_imports=True, encoding='latin1' )
        except FileNotFoundError:
            raise IOError ('Sleepdata file {} not found in {}'.format(filename, self.directory))
        self.__dict__.update(tmp_dict)
        return True


    def save_object(self, filename = 'sleepdata.pkl', path = None):
        """
        restores a previously stored SleepData object
        """
        if not filename [-4:] == '.pkl': filename = filename + '.pkl'
        if path == None: path = self.directory
        print('Saving data at {}'.format(filename))
        with open(os.path.join(path, filename), 'wb') as f:
            cPickle.dump(self.__dict__,f,2)
            
    
    def load_hypno_(self, files):
        self.hypno = []
        self.hypno_files = files
        for f in files:
            hypno  = self.load_hypnogram(os.path.join(self.directory + f), mode = 'overwrite')
            self.hypno.append(hypno)
            
            
    def _progress(self, description):
        self.tqdmloop.set_description(description + ' ' * (10-len(description)))
        self.tqdmloop.refresh()
        
    def _print(self, statement):
        if self.tqdmloop:
            self.tqdmloop.write(str(statement))
            self.tqdmloop.refresh()
        else:
            print(statement)


    def load_eeg_hypno(self, eeg_file, hypno_file, chuck_size = 3000, resampling = True, mode = 'standard', pool=False):
        """
        :param filename: loads the given eeg file
        :param mode: mode fro loading hypno-file
        """
        self._progress('Loading')
        if not pool: pool = Pool(3)
        self.resample = resampling
        hypno  = self.load_hypnogram(os.path.join(self.directory, hypno_file), mode = mode)
        header = self.load_eeg_header(os.path.join(self.directory, eeg_file), verbose='WARNING', preload=True)
        if self.channels['EEG'] == False: self.channels['EEG'], self.references['RefEEG'] = self.infer_channels(header.ch_names, 'EEG')
        if self.channels['EMG'] == False: self.channels['EMG'], self.references['RefEMG'] = self.infer_channels(header.ch_names, 'EMG')
        if self.channels['EOG'] == False: self.channels['EOG'], self.references['RefEOG'] = self.infer_channels(header.ch_names, 'EOG')
        self.available_channels = header.ch_names
        self.sfreq = np.round(header.info['sfreq'])
        self.check_for_normalization(header)

        picks, labels = self.check_channels(header)
        data,_ = deepcopy(header[picks, :])
#        print('setup: ', self.channels, self.references)
        eeg = data[labels.index('EEG'),:]
        if self.references['RefEEG']:
            eeg = eeg - data[labels.index('RefEEG'),:]
        emg = data[labels.index('EMG'),:]
        if self.references['RefEMG']: 
            emg = emg - data[labels.index('RefEMG'),:]          
        eog = data[labels.index('EOG'),:]
        if self.references['RefEOG']:
            eog = eog - data[labels.index('RefEOG'),:]
   
        self._progress('Filtering' )
        eeg = butter_bandpass_filter(eeg, 0.15, self.sfreq)
        emg = butter_bandpass_filter(emg, 10.0, self.sfreq)
        eog = butter_bandpass_filter(eog, 0.15, self.sfreq)
        # Resampling
        
        if not np.isclose(self.sfreq, 100):
            if resampling == True:
                self._progress('Resampling' )
                res_eeg = pool.apply_async(mne.io.RawArray(np.stack([eeg]), mne.create_info(1, self.sfreq, 'eeg'), verbose=0).resample, args = (100.,))
                res_emg = pool.apply_async(mne.io.RawArray(np.stack([emg]), mne.create_info(1, self.sfreq, 'eeg'), verbose=0).resample, args = (100.,))
                res_eog = pool.apply_async(mne.io.RawArray(np.stack([eog]), mne.create_info(1, self.sfreq, 'eeg'), verbose=0).resample, args = (100.,))
                eeg,_ = res_eeg.get(timeout=30)[0,:]
                emg,_ = res_emg.get(timeout=30)[0,:]
                eog,_ = res_eog.get(timeout=30)[0,:]
                eeg =eeg[0,:]
                emg =emg[0,:]
                eog =eog[0,:]
                self.sfreq = 100
            else:
                self._print ('Not resampling')
        self._progress('Loading')
        signal = np.stack([eeg,emg,eog]).swapaxes(0,1)
        hypno_len = len(hypno)
        eeg_len   = len(signal)
#        print('length: hypno {} eeg {}'.format(hypno_len, eeg_len))
        epoch_len = int(eeg_len / hypno_len / self.sfreq) 
        self.samples_per_epoch = int(epoch_len * self.sfreq) 
        trunc_len = (len(signal)//self.samples_per_epoch)*self.samples_per_epoch
        signal = signal[:trunc_len]     # remove left over to ensure len(data)%3000==0
        
        return signal.astype(self.dtype), hypno
        
    def check_data(self):
        """
        checks if data is in good shape
        1. Checks if there are missing segments (signal == 0 for whole epoch)
        2. Checks if the labels are all there (TODO)
        """
        for i, subject  in enumerate(self.data):
            epochs      = subject.reshape([-1, self.chunk_len, subject.shape[-1]])
            iszero      = np.isclose(epochs, 0, atol=0.0005)
            epochs_zero = np.mean(iszero, axis=1)
            epochs_zero = np.max(epochs_zero,1)
            if np.any(epochs_zero == 1):
                where = np.where(epochs_zero==1)[0]
                print('WARNING: Missing data in {} epoch{} ({:.1f}%) of subject {} (file: {})/n'.format(len(where), 's' if len(where)>1 else '',np.mean(epochs_zero)*100,i, self.eeg_files[i]))
            
        
    def shuffle_data(self):
        """
        Shuffle subjects that are loaded. Returns None
        """
        print('DEPRECATED: Please do not shuffle inside the sleeploader')
        if self.loaded == False: print('ERROR: Data not yet loaded')
        self.data, self.hypno, self.shuffle_index, self.subjects = shuffle(self.data, self.hypno, self.shuffle_index, self.subjects, random_state=self.rng)
        return None
        
        
    def get_subject(self, index):  
        """
        :param index: get subject [index] from loaded data. indexing from before shuffle is preserved
        """
        if self.loaded == False: print('ERROR: Data not loaded yet')
        return self.data[self.shuffle_index.index(index)], self.hypno[self.shuffle_index.index(index)] # index.index(index), beautiful isn't it?? :)
        
    
    def get_all_data(self, flat=True, groups = False):
        """
        returns all data that is loaded
        :param flat: select if data will be returned in a flat list or a list per subject
        """
    
        if self.loaded == False: print('ERROR: Data not loaded yet')
            
        if flat == True:
            return  self._makeflat(groups=groups)
        else:
            return self.data, self.hypno
        
        
    def _makeflat(self, start=None, end=None, groups = False):     
        eeg = list()
        for sub in self.data[start:end]:
            if len(sub) % self.chunk_len == 0:
                eeg.append(sub.reshape([-1, self.chunk_len,3]))
            else:
                print('ERROR: Please choose a chunk length that is a factor of {}'.format(self.samples_per_epoch))
                return [0,0]
        hypno = list()
        group = list()
        hypno_repeat = self.samples_per_epoch / self.chunk_len
        idx = 0
        for sub in self.hypno[start:end]:
            hypno.append(np.repeat(sub, hypno_repeat))
            group.append(np.repeat(idx, len(hypno[-1])))
            idx += 1
        
        if groups:
            return np.vstack(eeg), np.hstack(hypno), np.hstack(group)
        else:
            return np.vstack(eeg), np.hstack(hypno)
       
        
    def load(self, sel = [], channels = None, references = None, resampling = True, chunk_len = 3000, 
             flat = None, force_reload = False, shuffle = False, dtype=np.float32):
        """
        :param sel:          np.array with indices of files to load from the directory. Natural sorting.
        :param channels:     dict with form 'EEG':'channel_name', which channel to use for which modality (EEG,EMG,EOG). If none, will try to infer automatically
        :param reference:    dict with form 'EEG':'channel_name', which channel to use as reference. If None, no rereferencing will be applied
        :param shuffle:      shuffle subjects or not
        :param force_reload: reload data even if already loaded
        :param flat:         select if data will be returned in a flat array or a list per subject
        :param flat:         select if data will be returned in a flat array or a list per subject
        """
        if channels is not None: self.channels = channels
        if references is not None: self.references = references

        self.chunk_len = chunk_len        
        if self.loaded == True and force_reload == False and np.array_equal(sel, self.selection)==True:
            print('Getting Dataset')
            if shuffle == True:
                self.shuffle_data()
            if flat == True:
                return self._makeflat()
            elif flat == False:
                return self.data,self.hypno   
            else:
                print('No return mode set. Just setting new chunk_len')
                return
            
        elif force_reload==True:
            print('Reloading Dataset')
            
        else:
            print('Loading Dataset') 
            
        self.dtype = dtype   
        self.data = list()
        self.hypno = list()  
        self.selection = sel    
        self.rng = random.RandomState(seed=23)
    
        # check hypno_filenames
        self.hypno_files = [s for s in os.listdir(self.directory) if s.endswith('.txt')]
        self.hypno_files = sorted(self.hypno_files, key = natural_key)

        # check eeg_filenames
        self.eeg_files = [s for s in os.listdir(self.directory) if s.endswith(('.vhdr', 'edf'))]
        self.eeg_files = sorted(self.eeg_files, key = natural_key)
        
        if len(self.hypno_files)  != len(self.eeg_files): 
            print('ERROR: Not the same number of Hypno and EEG files. Hypno: ' + str(len(self.hypno_files))+ ', EEG: ' + str(len(self.eeg_files)))
            
        # select slice
        if sel==[]: sel = range(len(self.eeg_files))
        self.hypno_files = list(map(self.hypno_files.__getitem__, sel))
        self.eeg_files   = list(map(self.eeg_files.__getitem__, sel))
        self.shuffle_index = list(sel);
        self.subjects = zip(self.eeg_files,self.hypno_files)

        # load EEG and adapt Hypno files
        self.tqdmloop = trange(len(self.eeg_files), desc='Loading data')
        with Pool(3) as p:
            for i in self.tqdmloop:
                eeg, curr_hypno = self.load_eeg_hypno(self.eeg_files[i], self.hypno_files[i], chunk_len, resampling, pool=p)
                if(len(eeg) != len(curr_hypno) * self.samples_per_epoch):
                    self._print('WARNING: EEG epochs and Hypno epochs have different length {}:{} in {}.'.format(len(eeg),len(curr_hypno)* self.samples_per_epoch,self.eeg_files[i]))
                    if len(eeg) > len(curr_hypno) * self.samples_per_epoch:
                        self._print('Truncating EEG')
                        eeg = eeg[:len(curr_hypno) * self.samples_per_epoch]
                self.data.append(eeg)
                self.hypno.append(curr_hypno)
        self.tqdmloop = False    
        self.loaded = True
        
        # shuffle if wanted
        if shuffle == True:
            self.shuffle_data()
            
        # select if data will be returned in a flat array or a list per subject
        if flat == True:
            return  self._makeflat()
        elif flat == False:
            return self.data,self.hypno
            
#print('loaded sleeploader.py', __name__)



