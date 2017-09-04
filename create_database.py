# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 15:28:04 2017

@author: Simon
"""
#create sleepdata objects

if __name__ == '__main__': 
    #%%
    import sleeploader
    import tools
    import numpy as np
    import gc
        #%%

    sleep = sleeploader.SleepDataset('C:\sleep\cshs50')
    sleep.load_object()
#    sleep.save_object()
    data, target, groups = sleep.get_all_data(groups=True)
    
    feats_eeg = tools.feat_eeg(data[:,:,0])
    feats_emg = tools.feat_emg(data[:,:,1])
    feats_eog = tools.feat_eog(data[:,:,2])
    np.save('target.npy', target)
    np.save('groups.npy', groups)
    np.save('feats_eeg.npy', feats_eeg)
    np.save('feats_eog.npy', feats_eog)
    np.save('feats_emg.npy', feats_emg)
    del data, target, groups
    del feats_eeg, feats_emg, feats_eog
    del sleep; gc.collect()


    #%%
    sleep = sleeploader.SleepDataset('C:\sleep\emsa')
    sleep.load(list(range(14))+list(range(32,51)))
    sleep.save_object()
    del sleep; gc.collect()

    sleep = sleeploader.SleepDataset('C:\sleep\emsa')
    sleep.load(list(range(14,32)))
    sleep.save_object('children')   
    del sleep; gc.collect()

    
    #%%
    sleep = sleeploader.SleepDataset('C:/sleep/cshsh100')
    sleep.load()
    sleep.save_object()
    del sleep; gc.collect()
    
    #%%
    sleep = sleeploader.SleepDataset('C:/sleep/vinc')
    sleep.load()
    sleep.save_object()
    del sleep; gc.collect()    
    
    #%%
    sleep = sleeploader.SleepDataset('C:/sleep/edfx')
    sleep.load()
    sleep.save_object()
    data = sleep.data
    hypno = sleep.hypno
    new_data = []
    new_hypno = []
    for d, h in zip(data,hypno):
        if d.shape[0]%3000!=0: d = d[:len(d)-d.shape[0]%3000]
        d = d.reshape([-1,3000,3])
        if 9 in h:
            delete = np.where(h==9)[0]
    #        if len(delete)>0:print('deleting {} epochs'.format(len(delete)))
            d = np.delete(d, delete, 0)
            h = np.delete(h, delete, 0)
        begin = np.where(h-np.roll(h,1)!=0)[0][0]-300
        end = np.where(h-np.roll(h,1)!=0)[0][-1]+30
        
        d = d[begin:end]
        h = h[begin:end]
    
        d = d.ravel()
        new_hypno.append(h)
        new_data.append(d*1000000)
        print (len(h), len(d)//3000/3)
    
        
    
    sleep.data = new_data
    sleep.hypno = new_hypno
    sleep.save_object()
    del sleep; gc.collect()